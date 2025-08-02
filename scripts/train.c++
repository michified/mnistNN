#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#define endl "\n"
#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")
using namespace std;

const int RES = 28;
const int NUMTRAIN = 12000;
const int TRAINFILES = 5;
const int NUMTEST = 10000;
const int EPOCHS = 10;
const int BATCHSIZE = 500;
const int LAYERS = 2;
const int LAYERSIZE = 50;
const int OUTPUTS = 10;
const double LEARNINGRATE = 0.001;
const double DECAY1 = 0.99;
const double DECAY2 = 0.999;
const double DELTA = 0.00000001;
static random_device rd;
static mt19937 gen(rd());
normal_distribution<double> d(0, (double) 2 / (RES * RES));
auto rng = default_random_engine {rd()};

class Picture {
public:
    int label;
    double vals[RES * RES];
};

vector<Picture> trainData(NUMTRAIN * TRAINFILES), testData(NUMTEST);

void readData() {
    cout << "Reading data..." << endl;
    int i, k, tmp;
    string inp, val;
    for (int f = 0; f < TRAINFILES; f++) {
        ifstream cin("data/mnist_train" + to_string(f + 1) + ".txt");
        for (k = f * NUMTRAIN; k < (f + 1) * NUMTRAIN; k++) {
            cin >> inp;
            stringstream ss(inp);
            getline(ss, val, ',');
            trainData[k].label = stoi(val);
            for (i = 0; i < RES * RES; i++) {
                getline(ss, val, ',');
                trainData[k].vals[i] = (double) stoi(val);
            }
        }
        cin.close();
    }

    ifstream cin2("data/mnist_test.txt");
    for (k = 0; k < NUMTEST; k++) {
        cin2 >> inp;
        stringstream ss(inp);
        getline(ss, val, ',');
        testData[k].label = stoi(val);
        for (i = 0; i < RES * RES; i++) {
            getline(ss, val, ',');
            testData[k].vals[i] = (double) stoi(val);
        }
    }
    cin2.close();
    cout << "Finished reading data." << endl;
}


class Neuron {
public:
    vector<pair<double, Neuron*>> connections;
    double bias, a, z;

    void init(int prevN) {
        connections.resize(prevN);
        for (auto& connection : connections) connection.first = d(gen);
        bias = 0;
    }

    double ReLU(double x) {
        return max(0.0, x);
    }

    void computeVal(bool clamp) {
        z = bias;
        for (auto& connection : connections) z += connection.first * connection.second->a;
        a = clamp ? ReLU(z) : z;
    }
};

class Layer {
public:
    vector<Neuron> neurons;

    void init(int n, int prevN) {
        neurons.resize(n);
        for (int i = 0; i < n; i++) neurons[i].init(prevN);
    }
};

vector<Layer> layers(LAYERS + 2);

void initLayers() {
    cout << "Creating the neural net..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    int prev = RES * RES, i, j;
    layers[0].init(prev, 0);
    for (i = 1; i <= LAYERS; i++) {
        layers[i].init(LAYERSIZE, prev);
        for (auto& neuron : layers[i].neurons) {
            for (j = 0; j < prev; j++) neuron.connections[j].second = &layers[i - 1].neurons[j];
        }
        prev = LAYERSIZE;
    }
    layers[LAYERS + 1].init(OUTPUTS, prev);
    for (auto& neuron : layers[LAYERS + 1].neurons) {
        for (j = 0; j < prev; j++) neuron.connections[j].second = &layers[LAYERS].neurons[j];
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Neural net created in " << duration.count() << " milliseconds." << endl;
}

double dReLU(double x) {
    return x > 0;
}

vector<double> softmax() {
    vector<double> res(OUTPUTS);
    double tot = 0;
    for (int i = 0; i < OUTPUTS; i++) tot += exp(layers[LAYERS + 1].neurons[i].z);
    for (int i = 0; i < OUTPUTS; i++) res[i] = exp(layers[LAYERS + 1].neurons[i].z) / tot;
    return res;
}

vector<pair<vector<int>, double>> connectionGrads; // layerR, neuronL, neuronR, change
vector<pair<pair<int, int>, double>> biasGrads; // layer, neuron, change
vector<pair<double, double>> connectionGradsAdam, biasGradsAdam; // mean, var
int tCG, tBG;
double decay1PowT = 1, decay2PowT = 1;

void backpropagate(int layer, vector<double> d) {
    int prev, neuron;
    for (neuron = 0; neuron < (layer <= LAYERS ? LAYERSIZE : OUTPUTS); neuron++) {
        for (prev = 0; prev < (layer > 1 ? LAYERSIZE : RES * RES); prev++) {
            if (tCG == connectionGrads.size()) connectionGrads.push_back({{layer, prev, neuron}, 0.0});
            if (tCG == connectionGradsAdam.size()) connectionGradsAdam.push_back({0.0, 0.0});
            connectionGrads[tCG].second += d[neuron] * layers[layer - 1].neurons[prev].a;
            tCG++;
        }
        if (tBG == biasGrads.size()) biasGrads.push_back({{layer, neuron}, 0.0});
        if (tBG == biasGradsAdam.size()) biasGradsAdam.push_back({0.0, 0.0});
        biasGrads[tBG].second += d[neuron];
        tBG++;
    }
    if (layer == 1) return;
    vector<double> newd(LAYERSIZE);
    for (prev = 0; prev < (layer > 1 ? LAYERSIZE : RES * RES); prev++) {
        for (neuron = 0; neuron < (layer <= LAYERS ? LAYERSIZE : OUTPUTS); neuron++) {
            newd[prev] += d[neuron] * (layer == LAYERS + 1 ? 1 : dReLU(layers[layer].neurons[neuron].z)) * layers[layer].neurons[neuron].connections[prev].first;
        }
    }
    backpropagate(layer - 1, newd);
}

int computeBatch(bool train, int batchSize) {
    int i, j, k, correct = 0, hiLabel;
    double hi;
    double totCost = 0;
    Picture* picture;
    vector<double> d(OUTPUTS), softA;
    for (k = 0; k < batchSize; k++) {
        picture = train ? &trainData[k] : &testData[k];
        for (i = 0; i < RES * RES; i++) layers[0].neurons[i].a = picture->vals[i];
        for (i = 1; i <= LAYERS; i++) {
            for (j = 0; j < LAYERSIZE; j++) layers[i].neurons[j].computeVal(true);
        }
        hi = -1;
        hiLabel = 0;
        for (i = 0; i < OUTPUTS; i++) {
            layers[LAYERS + 1].neurons[i].computeVal(false);
        }
        softA = softmax();
        for (i = 0; i < OUTPUTS; i++) {
            if (train) d[i] = softA[i] - (picture->label == i);
            if (softA[i] > hi) {
                hi = softA[i];
                hiLabel = i;
            }
        }
        correct += hiLabel == picture->label;
        if (train) {
            tCG = 0;
            tBG = 0;
            backpropagate(LAYERS + 1, d);
        }
        fill(d.begin(), d.end(), 0);
    }
    return correct;
}

void trainNetwork() {
    cout << "Training the neural net..." << endl;
    auto totalStart = chrono::high_resolution_clock::now();
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        auto epochStart = chrono::high_resolution_clock::now();
        shuffle(trainData.begin(), trainData.end(), rng);
        double totCorrectTrain = 0, totCorrectTest = 0;
        for (int batch = 0; batch < NUMTRAIN * TRAINFILES / BATCHSIZE; batch++) {
            cout << batch << "\n";
            double biasCorrectedMean, biasCorrectedVar;
            totCorrectTrain += computeBatch(true, BATCHSIZE);
            for (int i = 0; i < connectionGrads.size(); i++) {
                auto& connectionGrad = connectionGrads[i];
                auto& connectionGradAdam = connectionGradsAdam[i];
                connectionGrad.second /= BATCHSIZE;
                connectionGradAdam.first = connectionGradAdam.first * DECAY1 + connectionGrad.second * (1 - DECAY1);
                connectionGradAdam.second = connectionGradAdam.second * DECAY2 + connectionGrad.second * connectionGrad.second * (1 - DECAY2);
                biasCorrectedMean = connectionGradAdam.first / (1 - decay1PowT);
                biasCorrectedVar = connectionGradAdam.second / (1 - decay2PowT);
                layers[connectionGrad.first[0]].neurons[connectionGrad.first[2]].connections[connectionGrad.first[1]].first -= LEARNINGRATE * (biasCorrectedMean / ((sqrt(biasCorrectedVar)) + DELTA));
                connectionGrad.second = 0;
            }
            for (int i = 0; i < biasGrads.size(); i++) {
                auto& biasGrad = biasGrads[i];
                auto& biasGradAdam = biasGradsAdam[i];
                biasGrad.second /= BATCHSIZE;
                biasGradAdam.first = biasGradAdam.first * DECAY1 + biasGrad.second * (1 - DECAY1);
                biasGradAdam.second = biasGradAdam.second * DECAY2 + biasGrad.second * biasGrad.second * (1 - DECAY2);
                biasCorrectedMean = biasGradAdam.first / (1 - decay1PowT);
                biasCorrectedVar = biasGradAdam.second / (1 - decay2PowT);
                layers[biasGrad.first.first].neurons[biasGrad.first.second].bias -= LEARNINGRATE * (biasCorrectedMean / ((sqrt(biasCorrectedVar)) + DELTA));
                biasGrad.second = 0;
            }
            rotate(trainData.begin(), trainData.begin() + BATCHSIZE, trainData.end());
        }
        decay1PowT *= DECAY1;
        decay2PowT *= DECAY2;
        // cout << "Testing..." << endl;
        totCorrectTest = computeBatch(false, NUMTEST);
        auto epochEnd = chrono::high_resolution_clock::now();
        auto epochDuration = chrono::duration_cast<chrono::seconds>(epochEnd - epochStart);
        
        cout << "Epoch #" + to_string(epoch) + " complete in " << epochDuration.count() << " seconds." << endl;
        cout << "Training data accuracy: " + to_string(totCorrectTrain / (NUMTRAIN * TRAINFILES) * 100).substr(0, 5) + "%." << endl;
        cout << "Testing data accuracy: " + to_string(totCorrectTest / NUMTEST * 100).substr(0, 5) + "%." << endl;
    }
    auto totalEnd = chrono::high_resolution_clock::now();
    auto totalDuration = chrono::duration_cast<chrono::seconds>(totalEnd - totalStart);
    cout << "Training complete in " << totalDuration.count() << " seconds." << endl;
}

void exportNetwork() {
    remove("model.txt");
    cout << "Exporting network..." << endl;
    ofstream cout2("model.txt");
    cout2 << LAYERS << ' ' << LAYERSIZE << endl;
    int i, j, k;
    for (i = 1; i <= LAYERS + 1; i++) {
        for (j = 0; j < (i == LAYERS + 1 ? OUTPUTS : LAYERSIZE); j++) {
            for (k = 0; k < (i == 1 ? RES * RES : LAYERSIZE); k++) {
                cout2 << fixed << setprecision(30) << layers[i].neurons[j].connections[k].first << endl;
            }
            cout2 << fixed << setprecision(30) << layers[i].neurons[j].bias << endl;
        }
    }
    cout2.close();
    cout << "Network exported." << endl;
}

int main() {
    // ios_base::sync_with_stdio(false);
    // cin.tie(nullptr);

    readData();
    initLayers();
    trainNetwork();
    exportNetwork();
    return 0;
}