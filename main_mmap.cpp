//Inspired from code from https://mecha-mind.medium.com/understanding-when-and-how-to-use-memory-mapped-files-b94707df30e9

#include <iostream>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <unordered_set>
#include <cstdlib>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

std::mt19937 rng(1000);
std::uniform_real_distribution<double> uniform(0.0, 1.0);

void handle(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}


#define MMAP_FILE "hnsw_index.bin"
#define MAX_MMAP_LENGTH 1024 * 1024 * 250  // 250MB

char *mmap_obj;
int fd;




void setup_mmap() {
    fd = open(MMAP_FILE, O_RDWR | O_CREAT, 0666);
    if (fd == -1) handle("open");
    struct stat sb;
    fstat(fd, &sb);
    if (sb.st_size == 0) {
        if (ftruncate(fd, MAX_MMAP_LENGTH) == -1) handle("ftruncate");
    }
    mmap_obj = (char *)mmap(NULL, MAX_MMAP_LENGTH, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmap_obj == MAP_FAILED) handle("mmap");
}



class HNSWNode {
    public:
        HNSWNode(int id, int level) : id(id), level(level) {}
        int getId() const { return id; }
        int getLevel() const { return level; }
        std::vector<float> getVector() const { return embedding; }
        void setVector(const std::vector<float>& vec) { embedding = vec; }
        std::vector<HNSWNode*> getNeighbors(int level) const {
            if (neighbors.find(level) != neighbors.end()) {
                return neighbors.at(level);
            }
            return {};
        }
        void addNeighbor(int level, HNSWNode* neighbor) {
            neighbors[level].push_back(neighbor);
        }
        int getNeighborsSize(int level) const {
            auto it = neighbors.find(level);
            if (it == neighbors.end()) return 0;
            return it->second.size();
        }
        int getIdx() const {
            return id;
        }
        void clearNeighbors(int level) {
            if (neighbors.find(level) != neighbors.end())
                neighbors[level].clear();
        }



    private:
        int id;
        int level;
        std::unordered_map<int, std::vector<HNSWNode*>> neighbors;
        std::vector<float> embedding;
};



class HNSW {
private:
    std::unordered_map<int, HNSWNode*> nodes;
    HNSWNode* entryPoint;
    int nodeId = 0;
    const int maxLevel;
    std::vector<int> maxM;
    int ef;
    double mL;

    int randomLevel() {
        double r = uniform(rng);
        int level = (int) std::floor(-std::log(r) * mL);
        return level;
    }

public:
    HNSW(int maxLevel, int ef, double mL) 
    : maxLevel(maxLevel), ef(ef), mL(mL) {
    entryPoint = nullptr;
    maxM = {32, 16, 16, 12, 8};

    }
    std::vector<HNSWNode*> getAllNodes() {
        std::vector<HNSWNode*> result;
        for (auto& kv : nodes) {
            result.push_back(kv.second);
        }
        return result;
    }

    std::vector<HNSWNode*> select_neighbors(const std::vector<float>& query, int k, const std::vector<HNSWNode*>& candidates) {
        std::vector<HNSWNode*> result;
        std::priority_queue<
            std::pair<float, HNSWNode*>,
            std::vector<std::pair<float, HNSWNode*>>,
            std::greater<std::pair<float, HNSWNode*>>
        > pq;

        for (auto candidate : candidates) {
            float dist = distance(query, candidate->getVector());
            pq.push(std::make_pair(dist, candidate));
            if (pq.size() > k) {
                pq.pop();
            }
        }
        while (!pq.empty()) {
            result.push_back(pq.top().second);
            pq.pop();
        }
        return result;
    }


    float distance(const std::vector<float>& a, const std::vector<float>& b) {
        float dist = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sqrt(dist);
    }


    std::vector<HNSWNode*> search_layer(const std::vector<float>& query, std::vector<HNSWNode*> ep, int ef, int level) {
        std::unordered_set<HNSWNode*> visited;
        std::priority_queue<std::pair<float, HNSWNode*>> result;
        std::priority_queue<
            std::pair<float, HNSWNode*>,
            std::vector<std::pair<float, HNSWNode*>>,
            std::greater<std::pair<float, HNSWNode*>>
        > candidates;

        for (auto node : ep) {
            float nodeDist = distance(query, node->getVector());
            visited.insert(node);
            candidates.push({nodeDist, node});
            result.push({nodeDist, node});
        }
    
        while (!candidates.empty()) {
            auto current = candidates.top().second;
            auto dist = candidates.top().first;
            candidates.pop();
    
            if (!result.empty() && result.top().first < dist) {
                break;
            }
    
            for (auto neighbor : current->getNeighbors(level)) {
                if (visited.find(neighbor) != visited.end()) continue;
                visited.insert(neighbor);
    
                float curDist = distance(query, neighbor->getVector());
                float farthest = result.empty() ? std::numeric_limits<float>::max() : result.top().first;
    
                if (result.size() >= (size_t)ef && curDist >= farthest) continue;

    
                candidates.push({curDist, neighbor});
                result.push({curDist, neighbor});
    
                if (result.size() > ef) result.pop();
            }
        }
    
        std::vector<HNSWNode*> ret;
        while (!result.empty()) {
            ret.push_back(result.top().second);
            result.pop();
        }
        std::reverse(ret.begin(), ret.end());
        return ret;

    }
    
    std::vector<HNSWNode*> knnSearch(const std::vector<float>& query, int k) {
        std::vector<HNSWNode*> ep;
        if (entryPoint != nullptr) {
            ep.push_back(entryPoint);
        }
        for (int lc = maxLevel; lc > 0; lc--) {
            auto w = search_layer(query, ep, 1, lc);
            if (w.size() > 0) {
                ep = {w[0]};
            }
        }
        auto w = search_layer(query, ep, k, 0);
        return w;

    }

    void insert(const std::vector<float>& query) {
        int level = randomLevel();
        HNSWNode* newNode = new HNSWNode(nodeId++, level);
        newNode->setVector(query);
        nodes[newNode->getId()] = newNode;
        if (entryPoint == nullptr) {
            entryPoint = newNode;
        } 
        std::priority_queue<
            std::pair<float, HNSWNode*>,
            std::vector<std::pair<float, HNSWNode*>>,
            std::greater<std::pair<float, HNSWNode*>> 
        > candidates;
        std::vector<HNSWNode*> ep;
        ep.push_back(entryPoint);
        for (int lc = maxLevel; lc > level; lc--) {
            auto w = search_layer(query, ep, 1, lc);
            if (w.size() > 0) {
                ep = {w[0]};
            }
        }
        for (int lc = std::min(maxLevel,level); lc >= 0; lc--) {
            auto w = search_layer(query, ep, ef, lc);
            auto neighbors = select_neighbors(query, maxM[lc], w);
            for (auto neighbor : neighbors) {
                newNode->addNeighbor(lc, neighbor);
                neighbor->addNeighbor(lc, newNode);
            }
            for (auto neighbor: neighbors) {
                if (neighbor->getNeighborsSize(lc) > maxM[lc]) {
                    auto newNeighbors = select_neighbors(neighbor->getVector(), maxM[lc], neighbor->getNeighbors(lc));
                    neighbor->clearNeighbors(lc);
                    for (auto newNeighbor : newNeighbors) {
                        neighbor->addNeighbor(lc, newNeighbor);
                    }
                }
                
            }
            ep = w;
        }
        if (level > maxLevel) {
            entryPoint = newNode;
        }
    }
};



void write_index_to_mmap(HNSW &hnsw) {
    std::vector<HNSWNode*> nodes = hnsw.getAllNodes();
    char *ptr = mmap_obj;  // Start writing from mmap memory
    size_t numNodes = nodes.size();
    memcpy(ptr, &numNodes, sizeof(size_t)); 
    ptr += sizeof(size_t);
    for (HNSWNode *node : nodes) {
        int id = node->getId();
        int level = node->getLevel();
        std::vector<float> embedding = node->getVector();
        memcpy(ptr, &id, sizeof(int));
        ptr += sizeof(int);
        memcpy(ptr, &level, sizeof(int));
        ptr += sizeof(int);
        size_t vecSize = embedding.size();
        memcpy(ptr, &vecSize, sizeof(size_t));
        ptr += sizeof(size_t);
        memcpy(ptr, embedding.data(), vecSize * sizeof(float));
        ptr += vecSize * sizeof(float);



        for (int l = 0; l <= level; l++) {
            std::vector<HNSWNode*> neighbors = node->getNeighbors(l);
            size_t numNeighbors = neighbors.size();
            memcpy(ptr, &numNeighbors, sizeof(size_t));
            ptr += sizeof(size_t);
            for (HNSWNode *neighbor : neighbors) {
                int neighborId = neighbor->getId();
                memcpy(ptr, &neighborId, sizeof(int));
                ptr += sizeof(int);
            }
        }
    }
}



void read_index_from_mmap(HNSW &hnsw) {
    char *ptr = mmap_obj;
    size_t numNodes;
    memcpy(&numNodes, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    std::unordered_map<int, HNSWNode*> nodeMap;

    for (size_t i = 0; i < numNodes; i++) {
        int id, level;
        memcpy(&id, ptr, sizeof(int));
        ptr += sizeof(int);
        memcpy(&level, ptr, sizeof(int));
        ptr += sizeof(int);
        size_t vecSize;
        memcpy(&vecSize, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        std::vector<float> embedding(vecSize);
        memcpy(embedding.data(), ptr, vecSize * sizeof(float));
        ptr += vecSize * sizeof(float);


        HNSWNode *node = new HNSWNode(id, level);
        node->setVector(embedding);
        nodeMap[id] = node;
        hnsw.insert(node->getVector());
    }

    for (size_t i = 0; i < numNodes; i++) {
        HNSWNode *node = nodeMap[i];

        for (int l = 0; l <= node->getLevel(); l++) {
            size_t numNeighbors;
            memcpy(&numNeighbors, ptr, sizeof(size_t));
            ptr += sizeof(size_t);

            for (size_t j = 0; j < numNeighbors; j++) {
                int neighborId;
                memcpy(&neighborId, ptr, sizeof(int));
                ptr += sizeof(int);

                if (nodeMap.find(neighborId) != nodeMap.end()) {
                    node->addNeighbor(l, nodeMap[neighborId]);
                }
            }
        }
    }
}



int main() {
    unlink(MMAP_FILE);
    setup_mmap();
    HNSW hnsw(5, 50, 0.3);
    HNSW hnsw_ram(5, 50, 0.3);
    std::cout << "Building new HNSW index..." << std::endl;
    auto start_insertion = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        std::vector<float> vec(384);
        for (int j = 0; j < 384; j++) {
            vec[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        hnsw.insert(vec);
        hnsw_ram.insert(vec);
    }
    auto end_insertion = std::chrono::high_resolution_clock::now();



    std::chrono::duration<double> insertionTime = end_insertion - start_insertion;
    std::cout << "HNSW insertion time (1000 vectors): " << insertionTime.count() << " seconds" << std::endl;
    std::cout << "Writing index to mmap..." << std::endl;
    auto start_write = std::chrono::high_resolution_clock::now();
    write_index_to_mmap(hnsw);
    auto end_write = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> writeTime = end_write - start_write;
    std::cout << "HNSW index write time: " << writeTime.count() << " seconds" << std::endl;
    std::vector<float> query(384);
    for (int j = 0; j < 384; j++) {
        query[j] = static_cast<float>(rand()) / RAND_MAX;
    }


    std::cout << "Performing k-NN search (mmap)..." << std::endl;
    auto start_hnsw = std::chrono::high_resolution_clock::now();
    std::vector<HNSWNode*> hnswResults = hnsw.knnSearch(query, 10);
    auto end_hnsw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hnswTime = end_hnsw - start_hnsw;


    std::cout << "HNSW search time (mmap): " << hnswTime.count() << " seconds" << std::endl;
    std::cout << "Performing k-NN search (RAM)..." << std::endl;
    auto start_hnsw_ram = std::chrono::high_resolution_clock::now();
    std::vector<HNSWNode*> hnswResultsRam = hnsw_ram.knnSearch(query, 10);
    auto end_hnsw_ram = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hnswTimeRam = end_hnsw_ram - start_hnsw_ram;


    std::cout << "HNSW search time (RAM): " << hnswTimeRam.count() << " seconds" << std::endl;
    close(fd);
    return 0;
}
