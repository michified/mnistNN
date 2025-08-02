#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#pragma GCC optimize("Ofast")
#define endl "\n"
using namespace std;

const int RES = 28;
const int NUMTRAIN = 12000;
const int TRAINFILES = 5;
const int NUMTEST = 10000;
const int EPOCHS = 50;
const int BATCHSIZE = 500;
const int LAYERS = 2;
const int LAYERSIZE = 50;
const int OUTPUTS = 10;
const double LEARNINGRATE = 0.0002;
const double DECAY1 = 0.99;
const double DECAY2 = 0.999;
const double DELTA = 0.00000001;
static random_device rd;
static mt19937 gen(rd());
auto rng = default_random_engine {rd()};

class Picture {
public:
    int label;
    double vals[RES * RES];
};

vector<Picture> trainData, testData;

void readData() {
    cout << "Reading data..." << endl;
    trainData.resize(NUMTRAIN * TRAINFILES);
    testData.resize(NUMTEST);
    
    int i, k;
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
                trainData[k].vals[i] = stoi(val) / 255.0;  // Normalize input
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
            testData[k].vals[i] = stoi(val) / 255.0;
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
        double stddev = sqrt(2.0 / prevN);
        normal_distribution<double> d(0, stddev);
        for (auto& connection : connections) {
            connection.first = d(gen);
        }
        bias = 0.01; 
    }

    double ReLU(double x) {
        return max(0.0, x);
    }

    void computeVal(bool applyActivation) {
        z = bias;
        for (auto& connection : connections) {
            z += connection.first * connection.second->a;
        }
        a = applyActivation ? ReLU(z) : z;
    }
};

class Layer {
public:
    vector<Neuron> neurons;

    void init(int n, int prevN) {
        neurons.resize(n);
        for (int i = 0; i < n; i++) {
            neurons[i].init(prevN);
        }
    }
};

vector<Layer> layers(LAYERS + 2);

void initLayers() {
    cout << "Creating the neural net..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    int prev = RES * RES;
    layers[0].init(prev, 0);
    
    for (int i = 1; i <= LAYERS; i++) {
        layers[i].init(LAYERSIZE, prev);
        for (int j = 0; j < LAYERSIZE; j++) {
            for (int k = 0; k < prev; k++) {
                layers[i].neurons[j].connections[k].second = &layers[i - 1].neurons[k];
            }
        }
        prev = LAYERSIZE;
    }
    
    layers[LAYERS + 1].init(OUTPUTS, prev);
    for (int j = 0; j < OUTPUTS; j++) {
        for (int k = 0; k < prev; k++) {
            layers[LAYERS + 1].neurons[j].connections[k].second = &layers[LAYERS].neurons[k];
        }
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Neural net created in " << duration.count() << " milliseconds." << endl;
}

double dReLU(double x) {
    return x > 0 ? 1.0 : 0.0;
}

vector<double> softmax() {
    vector<double> res(OUTPUTS);
    double max_val = *max_element(&layers[LAYERS + 1].neurons[0].z, &layers[LAYERS + 1].neurons[OUTPUTS].z);
    double tot = 0;
    for (int i = 0; i < OUTPUTS; i++) {
        res[i] = exp(layers[LAYERS + 1].neurons[i].z - max_val);
        tot += res[i];
    }
    for (int i = 0; i < OUTPUTS; i++) {
        res[i] /= tot;
    }
    return res;
}

vector<vector<vector<double>>> weightGrads;
vector<vector<double>> biasGrads;
vector<vector<vector<double>>> weightAdamM, weightAdamV;
vector<vector<double>> biasAdamM, biasAdamV;

void initializeGradStorage() {
    weightGrads.resize(LAYERS + 2);
    biasGrads.resize(LAYERS + 2);
    weightAdamM.resize(LAYERS + 2);
    weightAdamV.resize(LAYERS + 2);
    biasAdamM.resize(LAYERS + 2);
    biasAdamV.resize(LAYERS + 2);
    
    for (int l = 1; l <= LAYERS + 1; l++) {
        int currentSize = (l <= LAYERS) ? LAYERSIZE : OUTPUTS;
        int prevSize = (l == 1) ? RES * RES : LAYERSIZE;
        
        weightGrads[l].resize(currentSize, vector<double>(prevSize, 0.0));
        biasGrads[l].resize(currentSize, 0.0);
        weightAdamM[l].resize(currentSize, vector<double>(prevSize, 0.0));
        weightAdamV[l].resize(currentSize, vector<double>(prevSize, 0.0));
        biasAdamM[l].resize(currentSize, 0.0);
        biasAdamV[l].resize(currentSize, 0.0);
    }
}

void backpropagate(int layer, const vector<double>& d) {
    int currentSize = (layer <= LAYERS) ? LAYERSIZE : OUTPUTS;
    int prevSize = (layer == 1) ? RES * RES : LAYERSIZE;
    
    for (int n = 0; n < currentSize; n++) {
        biasGrads[layer][n] += d[n];
        for (int p = 0; p < prevSize; p++) {
            weightGrads[layer][n][p] += d[n] * layers[layer - 1].neurons[p].a;
        }
    }
    
    if (layer == 1) return;

    vector<double> newd(prevSize, 0.0);
    for (int p = 0; p < prevSize; p++) {
        for (int n = 0; n < currentSize; n++) {
            double grad_factor = (layer - 1 == 0) ? 1.0 : dReLU(layers[layer - 1].neurons[p].z);
            newd[p] += d[n] * layers[layer].neurons[n].connections[p].first * grad_factor;
        }
    }
    
    backpropagate(layer - 1, newd);
}

int computeBatch(bool train, int batchSize, int start_idx) {
    int correct = 0;
    vector<double> d(OUTPUTS, 0.0);

    for (int k = 0; k < batchSize; k++) {
        Picture* picture = &trainData[start_idx + k];

        for (int i = 0; i < RES * RES; i++) {
            layers[0].neurons[i].a = picture->vals[i];
        }
        
        for (int l = 1; l <= LAYERS; l++) {
            for (int n = 0; n < LAYERSIZE; n++) {
                layers[l].neurons[n].computeVal(true);
            }
        }
        
        for (int i = 0; i < OUTPUTS; i++) {
            layers[LAYERS + 1].neurons[i].computeVal(false);
        }
        
        vector<double> softA = softmax();
        int prediction = 0;
        double max_prob = softA[0];
        for (int i = 1; i < OUTPUTS; i++) {
            if (softA[i] > max_prob) {
                max_prob = softA[i];
                prediction = i;
            }
        }
        
        correct += (prediction == picture->label);
        
        if (train) {
            for (int i = 0; i < OUTPUTS; i++) {
                d[i] = softA[i] - (picture->label == i ? 1.0 : 0.0);
            }
            
            for (int l = 1; l <= LAYERS + 1; l++) {
                fill(biasGrads[l].begin(), biasGrads[l].end(), 0.0);
                for (auto& wg : weightGrads[l]) {
                    fill(wg.begin(), wg.end(), 0.0);
                }
            }

            backpropagate(LAYERS + 1, d);
        }
    }
    return correct;
}

void trainNetwork() {
    cout << "Training the neural net..." << endl;
    auto totalStart = chrono::high_resolution_clock::now();
    int totalTrain = NUMTRAIN * TRAINFILES;
    int totalBatches = totalTrain / BATCHSIZE;
    
    double decay1T = DECAY1;
    double decay2T = DECAY2;
    
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        auto epochStart = chrono::high_resolution_clock::now();
        shuffle(trainData.begin(), trainData.end(), rng);
        
        double totCorrectTrain = 0;
        double totCorrectTest = 0;
        
        for (int batch = 0; batch < totalBatches; batch++) {
            int start_idx = batch * BATCHSIZE;
            totCorrectTrain += computeBatch(true, BATCHSIZE, start_idx);

            for (int l = 1; l <= LAYERS + 1; l++) {
                int currentSize = (l <= LAYERS) ? LAYERSIZE : OUTPUTS;
                int prevSize = (l == 1) ? RES * RES : LAYERSIZE;
                
                for (int n = 0; n < currentSize; n++) {
                    double bg = biasGrads[l][n] / BATCHSIZE;
                    biasAdamM[l][n] = DECAY1 * biasAdamM[l][n] + (1 - DECAY1) * bg;
                    biasAdamV[l][n] = DECAY2 * biasAdamV[l][n] + (1 - DECAY2) * bg * bg;
                    
                    double m_hat = biasAdamM[l][n] / (1 - decay1T);
                    double v_hat = biasAdamV[l][n] / (1 - decay2T);
                    
                    layers[l].neurons[n].bias -= LEARNINGRATE * m_hat / (sqrt(v_hat) + DELTA);

                    for (int p = 0; p < prevSize; p++) {
                        double wg = weightGrads[l][n][p] / BATCHSIZE;
                        weightAdamM[l][n][p] = DECAY1 * weightAdamM[l][n][p] + (1 - DECAY1) * wg;
                        weightAdamV[l][n][p] = DECAY2 * weightAdamV[l][n][p] + (1 - DECAY2) * wg * wg;
                        
                        m_hat = weightAdamM[l][n][p] / (1 - decay1T);
                        v_hat = weightAdamV[l][n][p] / (1 - decay2T);
                        
                        layers[l].neurons[n].connections[p].first -= LEARNINGRATE * m_hat / (sqrt(v_hat) + DELTA);
                    }
                }
            }
            
            decay1T *= DECAY1;
            decay2T *= DECAY2;
        }

        totCorrectTest = computeBatch(false, NUMTEST, 0);
        
        auto epochEnd = chrono::high_resolution_clock::now();
        auto epochDuration = chrono::duration_cast<chrono::seconds>(epochEnd - epochStart);
        
        cout << "Epoch #" << epoch << " complete in " << epochDuration.count() << " seconds." << endl;
        cout << "Training data accuracy: " << fixed << setprecision(2) << (totCorrectTrain / totalTrain) * 100 << "%" << endl;
        cout << "Testing data accuracy: " << fixed << setprecision(2) << (totCorrectTest / (double)NUMTEST) * 100 << "%" << endl;
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
    
    for (int l = 1; l <= LAYERS + 1; l++) {
        int currentSize = (l == LAYERS + 1) ? OUTPUTS : LAYERSIZE;
        int prevSize = (l == 1) ? RES * RES : LAYERSIZE;
        
        for (int n = 0; n < currentSize; n++) {
            for (int p = 0; p < prevSize; p++) {
                cout2 << fixed << setprecision(15) << layers[l].neurons[n].connections[p].first << endl;
            }
            cout2 << fixed << setprecision(15) << layers[l].neurons[n].bias << endl;
        }
    }
    cout2.close();
    cout << "Network exported." << endl;
}

int main() {
    readData();
    initLayers();
    initializeGradStorage();  // Initialize gradient storage
    trainNetwork();
    exportNetwork();
    return 0;
}