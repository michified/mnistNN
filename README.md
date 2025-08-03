# Handwritten Digit Classifier in C++
**A from-scratch implementation achieving ~90% accuracy on MNIST**

---

## 🧠 Model Architecture
- **Input Layer**: 784 input neurons for the 28x28 image size for the MNIST dataset
- **Hidden Layers**: 2 hidden, fully connected layers, each with 50 neurons
- **Output Layer**: 10 output neurons corresponding with the digits 0~9, uses softmax activation

### Weight Count: 42310
| Input → Hidden Layer 1  | Hidden Layer 1 → 2 | Hidden Layer 2 → Output Layer | Bias Weights on Neurons | Total |
|:-------------------:|:----------------:|:-------------------------:|:-------------------:|:------:|
|   784×50 = 39,200   |  50×50 = 2,500  |        50×10 = 500        |    50+50+10 = 110   | 42,310 |

### Hyperparameters
| Hyperparameter | Value |
|:--------------:|:-----:|
| Epochs | 50 |
| Learning Rate | 2e-4 |
| $\beta_1$ (Adam) | 0.99 |
| $\beta_2$ (Adam) | 0.999 |

---

## 🚀 Features
- **Pure C++23** (no ML frameworks, check train.c++)
- **~100μs inference** per image (Intel i7-13700K)
- **Adam optimization** for higher accuracy
- **Visualization tools** written in python (display.py) to display and visualize prediction distributions and ground truths

---

## 📊 Statistics (ran on 13th Gen Intel(R) Core(TM) i7-13700K)
- **Neural Net Creation**: 5 ms
- **Training Time**: 1009 seconds
- **Prediction Time**: 108 microsecond average
- **Accuracy**: 90-91%
- **Model Size**: 805KB

---

## 🔬 Technical Insights
1. **Architecture tradeoffs and optimization**: More layers and neurons doesn't necessarily result in a higher accuracy. Testing with 3+ FC layers with 100K+ weights only resulted in a prediction accuracy of ~84% but with triple the training time
2. **Watch out for math.** Spent two hours debugging train.c++ to find a math error of multiplying derivatives from the wrong layer during backpropagation
3. **Input validation**: It is good to check for whether the input is what is expected, such as verifying whether the number of weights and the neural network architecture is correct in test.c++ from what was trained in train.c++.
4. **Cross-language integration**: Displaying graphics with python is much easier than C++, and both programming languages have methods to invoke executable files (e.g. C++ has system("python filename.py")).

---

## 🔍 Future Work
- **Further Optimization**: Such as hyperparameter tuning or testing other adaptive learning methods other than Adam
- **Further Dataset Testing**: Train, test, and benchmark performance on other datasets such as FashionMNIST
- **Benchmarking Performance**: Compare performance with standard implementations from PyTorch or TensorFlow and identify shortcomings and possible optimization strategies
- **GPU Compatibility**: Experiemnt with GPU integration (e.g. parallelization) to boost training routines

---

## 🚀 Build and Run
1. Make a copy of this repo
2. Navigate to the scripts directory:
   ```bash
   cd scripts
   ```
3. Compile the training program:
   ```bash
   g++ train.c++ -o train.exe
   ```
4. Run the training program to train the model:
   ```bash
   ./train.exe
   ```
5. Compile the testing program:
   ```bash
   g++ test.c++ -o test.exe
   ```
6. Run the testing program to evaluate and visualize results:
   ```bash
   ./test.exe
   ```

### Dependencies
#### System Requirements
- Windows/Linux/macOS

#### Environment Requirements
- Python 3.12+
- NumPy package
- GCC 12+ (`-std=c++23`)
- CMake 3.15+ (optional, for building from source)

---

## 📖 References and Resources
- 3Blue1Brown's Deep Learning course, episodes 1-4. https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- Hojjat Khodabakhsh. (2018). MNIST Dataset. https://www.kaggle.com/datasets/hojjatk/mnist-dataset

- Adam optimization. https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

