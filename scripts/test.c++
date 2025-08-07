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

const int NUMTEST = 10000;
const int RES = 28, OUTPUTS = 10;
int LAYERS, LAYERSIZE;

class Picture {
public:
    int label;
    double vals[RES * RES];
};

vector<Picture> testData;

void readData() {
    ifstream cin2("data/emnist-digits-test.txt");
    cout << "Reading data..." << endl;
    testData.resize(NUMTEST);
    int i, k;
    string val, inp;
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
    }

    double ReLU(double x) {
        return max(0.0, x);
    }

    void computeVal(bool clamp) {
        z = bias;
        for (auto& connection : connections) 
            z += connection.first * connection.second->a;
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

vector<Layer> layers;

vector<double> softmax() {
    vector<double> res(OUTPUTS);
    double max_val = *max_element(&layers[LAYERS+1].neurons[0].z, 
                                 &layers[LAYERS+1].neurons[OUTPUTS].z);
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

void initLayers() {
    cout << "Creating the neural net..." << endl;
    ifstream cin2("model.txt");
    if (not cin2.is_open()) {
        cerr << "Error: Could not open model file!" << endl;
        exit(1);
    }

    cin2 >> LAYERS >> LAYERSIZE;
    layers.resize(LAYERS + 2);
    
    int prev = RES * RES;
    layers[0].init(prev, 0);
    
    for (int i = 1; i <= LAYERS; i++) {
        layers[i].init(LAYERSIZE, prev);
        for (auto& neuron : layers[i].neurons) {
            for (int j = 0; j < prev; j++) {
                neuron.connections[j].second = &layers[i-1].neurons[j];
            }
        }
        prev = LAYERSIZE;
    }
    
    layers[LAYERS + 1].init(OUTPUTS, prev);
    for (auto& neuron : layers[LAYERS + 1].neurons) {
        for (int j = 0; j < prev; j++) {
            neuron.connections[j].second = &layers[LAYERS].neurons[j];
        }
    }

    for (int l = 1; l <= LAYERS + 1; l++) {
        int currentSize = (l == LAYERS + 1) ? OUTPUTS : LAYERSIZE;
        int prevSize = (l == 1) ? RES * RES : LAYERSIZE;
        
        for (int n = 0; n < currentSize; n++) {
            for (int p = 0; p < prevSize; p++) {
                if (not (cin2 >> layers[l].neurons[n].connections[p].first)) {
                    cerr << "Error reading weights at layer " << l 
                         << ", neuron " << n << ", connection " << p << endl;
                    exit(1);
                }
            }

            if (not (cin2 >> layers[l].neurons[n].bias)) {
                cerr << "Error reading bias at layer " << l 
                     << ", neuron " << n << endl;
                exit(1);
            }
        }
    }
    
    string dummy;
    if (cin2 >> dummy) {
        cerr << "Warning: Extra data in model file!" << endl;
    }
    
    cin2.close();
    cout << "Network imported successfully." << endl;
}

void computeBatch() {
    cout << "Testing..." << endl;
    remove("preds.txt");
    ofstream cout2("preds.txt");
    int i, j, k, correct = 0;
    Picture* picture;
    
    auto totalStart = chrono::high_resolution_clock::now();
    long long totalPredictionTime = 0;
    for (k = 0; k < NUMTEST; k++) {
        picture = &testData[k];
        
        auto predStart = chrono::high_resolution_clock::now();
        
        for (i = 0; i < RES * RES; i++) 
            layers[0].neurons[i].a = picture->vals[i];
        
        for (i = 1; i <= LAYERS; i++) {
            for (j = 0; j < LAYERSIZE; j++) 
                layers[i].neurons[j].computeVal(true);
        }
        
        for (i = 0; i < OUTPUTS; i++) 
            layers[LAYERS + 1].neurons[i].computeVal(false);

        auto predictions = softmax();
        
        auto predEnd = chrono::high_resolution_clock::now();
        totalPredictionTime += chrono::duration_cast<chrono::microseconds>(predEnd - predStart).count();
        
        for (double conf : predictions) {
            cout2 << fixed << setprecision(4) << conf << ' ';
        }
        cout2 << endl;
        
        int prediction = 0;
        double max_conf = predictions[0];
        for (i = 1; i < OUTPUTS; i++) {
            if (predictions[i] > max_conf) {
                max_conf = predictions[i];
                prediction = i;
            }
        }
        correct += (prediction == picture->label);
    }
    cout2.close();
    
    auto totalEnd = chrono::high_resolution_clock::now();
    auto totalDuration = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart).count();
    double avgPredictionTime = static_cast<double>(totalPredictionTime) / NUMTEST;
    
    cout << "Testing finished in " << totalDuration << " ms" << endl;
    cout << "Average prediction time per image: " << avgPredictionTime << " microseconds" << endl;
    cout << "Accuracy: " << fixed << setprecision(2) 
         << (correct * 100.0 / NUMTEST) << "%" << endl;
}

int main() {
    readData();
    initLayers();
    computeBatch();
    system("python display.py");
    return 0;
}