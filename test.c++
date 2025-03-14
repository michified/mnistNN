#include <bits/stdc++.h>
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

Picture testData[NUMTEST];

void readData() {
    ifstream cin2("mnist_test.txt");
    cout << "Reading data..." << endl;
    int i, k;
    string val, inp;
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
    }

    double ReLU(double x) {
        return max(0.0, x);
    }

    void computeVal() {
        z = bias;
        for (auto& connection : connections) z += connection.first * connection.second->a;
        a = ReLU(z);
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

void initLayers() {
    cout << "Creating the neural net..." << endl;
    ifstream cin2("model.txt");
    cin2 >> LAYERS >> LAYERSIZE;
    layers.resize(LAYERS + 2);
    int prev = RES * RES, i, j, k;
    layers[0].init(prev, 0);
    for (i = 1; i <= LAYERS; i++) {
        layers[i].init(LAYERSIZE, prev);
        for (auto& neuron : layers[i].neurons) {
            for (j = 0; j < prev; j++) neuron.connections[j].second = &layers[i - 1].neurons[j];
        }
        prev = LAYERSIZE;
    }
    layers[LAYERS + 1].init(OUTPUTS, prev);
    for (auto& neuron : layers[i].neurons) {
        for (j = 0; j < prev; j++) neuron.connections[j].second = &layers[i - 1].neurons[j];
    }
    cout << "Neural net created." << endl;

    cout << "Importing network..." << endl;
    for (i = 1; i <= LAYERS + 1; i++) {
        for (j = 0; j < (i == LAYERS + 1 ? OUTPUTS : LAYERSIZE); j++) {
            for (k = 0; k < (i == 1 ? RES * RES : LAYERSIZE); k++) {
                cin2 >> layers[i].neurons[j].connections[k].first;
            }
            cin2 >> layers[i].neurons[j].bias;
        }
    }
    cin2.close();
    cout << "Network imported." << endl;
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

void computeBatch() {
    cout << "Testing..." << endl;
    remove("preds.txt");
    ofstream cout2("preds.txt");
    int i, j, k, correct = 0, hiLabel;
    double hi;
    double totCost = 0;
    Picture* picture;
    for (k = 0; k < NUMTEST; k++) {
        picture = &testData[k];
        for (i = 0; i < RES * RES; i++) layers[0].neurons[i].a = picture->vals[i];
        for (i = 1; i <= LAYERS; i++) {
            for (j = 0; j < LAYERSIZE; j++) layers[i].neurons[j].computeVal();
        }
        for (i = 0; i < OUTPUTS; i++) layers[LAYERS + 1].neurons[i].computeVal();
        for (double conf : softmax()) {
            cout2 << fixed << setprecision(2) << conf << ' ';
        }
        cout2 << endl;
    }
    cout2.close();
    cout << "Testing finished." << endl;
}

int main() {
    // ios_base::sync_with_stdio(false);
    // cin.tie(nullptr);

    readData();
    initLayers();
    computeBatch();
    system("display.py");
    return 0;
}