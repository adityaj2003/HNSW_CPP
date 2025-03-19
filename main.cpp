#include <iostream>
#include <string>
#include <queue>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <unordered_set>
#include <cstdlib>
#include <random>

std::mt19937 rng(1000);
std::uniform_real_distribution<double> uniform(0.0, 1.0);


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


int main() {
    const int numNodes = 1000;
    const int dim = 384;
    const int k = 10;
    const int maxLevel = 5;
    const int ef = 50;
    const double mL = 0.3;

    HNSW hnsw(maxLevel, ef, mL);

    auto start_insertion = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numNodes; i++) {
        std::vector<float> vec(dim);
        for (int j = 0; j < dim; j++) {
            vec[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        hnsw.insert(vec);
    }
    auto end_insertion = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> insertionTime = end_insertion - start_insertion;

    std::cout << "HNSW insertion time: " << insertionTime.count() << " seconds" << std::endl;


    std::vector<HNSWNode*> linearNodes = hnsw.getAllNodes();

    std::vector<float> query(dim);
    for (int j = 0; j < dim; j++) {
        query[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ---------- Linear Search ----------
    auto start_linear = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<float, HNSWNode*>> linearResults;
    for (auto node : linearNodes) {
        float d = hnsw.distance(query, node->getVector());
        linearResults.push_back({d, node});
    }
    std::sort(linearResults.begin(), linearResults.end(),
              [](auto a, auto b) { return a.first < b.first; });

    std::vector<HNSWNode*> linearK;
    std::vector<float> linearDistances;
    for (int i = 0; i < k && i < linearResults.size(); i++) {
        linearK.push_back(linearResults[i].second);
        linearDistances.push_back(linearResults[i].first);
    }
    auto end_linear = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> linearTime = end_linear - start_linear;

    auto start_hnsw = std::chrono::high_resolution_clock::now();
    std::vector<HNSWNode*> hnswResults = hnsw.knnSearch(query, k);
    auto end_hnsw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hnswTime = end_hnsw - start_hnsw;

    std::vector<float> hnswDistances;
    for (auto node : hnswResults) {
        hnswDistances.push_back(hnsw.distance(query, node->getVector()));
    }

    int matchCount = 0;
    for (auto node : hnswResults) {
        for (auto lnode : linearK) {
            if (node->getId() == lnode->getId()) {
                matchCount++;
                break;
            }
        }
    }

    // ---------- Output Results ----------
    std::cout << "Linear search time: " << linearTime.count() << " seconds" << std::endl;
    std::cout << "HNSW search time: " << hnswTime.count() << " seconds" << std::endl;
    std::cout << "Matching results: " << matchCount << " out of " << k << std::endl;

    std::cout << "Top 10 distances (Linear Search): ";
    for (float d : linearDistances) {
        std::cout << d << " ";
    }
    std::cout << std::endl;

    std::cout << "Top 10 distances (HNSW Search): ";
    for (float d : hnswDistances) {
        std::cout << d << " ";
    }

    std::cout << std::endl;
    std::cout << "Worst (max) distance (Linear Search): " << linearResults.back().first << std::endl;

    return 0;
}
