#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <limits>
#include <string>
#include <sstream>
#include <random>
#include <functional>
#include <iomanip>

// Класс для генерации точек 
class DataGenerator {
private:
    int classes;
    int dimensions;
    int* elements;
    int sum;
    std::ifstream fin;

    void generateData(std::ofstream& fout);

public:
    DataGenerator(const std::string& configFile);
    ~DataGenerator();

    void generateAndSave(const std::string& outputFile);
};

// Чтение данных с файла
DataGenerator::DataGenerator(const std::string& configFile) : classes(0), dimensions(0), elements(nullptr), sum(0), fin(configFile) {

    if (!fin.is_open()) {
        std::cerr << "Error opening config.txt" << std::endl;
        return;
    }

    fin >> classes;
    elements = new int[classes];
    for (int i = 0; i < classes; i++) {
        fin >> elements[i];
        sum += elements[i];
        if (sum < 0) {
            std::cerr << "\nOverflow: sum of elements is negative" << std::endl;
            delete[] elements;
            return;
        }
    }

    fin >> dimensions;
}

// Освобождение памяти
DataGenerator::~DataGenerator() {
    delete[] elements;
}

// Считываем диапазоны точек для каждого класса
void DataGenerator::generateData(std::ofstream& fout) {
    for (int i = 0; i < classes; i++) {
        for (int j = 0; j < dimensions; j++) {
            int min, max = 0;
            fin >> min;
            fin >> max;
            for (int k = 0; k < elements[i]; k++) {
                int value = min + rand() % (max - min + 1);
                fout << value << " ";
            }
            fout << std::endl;
        }
        fout << std::endl;
    }
}

// Вывод сгенерированных точек в новый файл
void DataGenerator::generateAndSave(const std::string& outputFile) {
    std::ofstream fout(outputFile);
    if (!fout.is_open()) {
        std::cerr << "Error opening " << outputFile << std::endl;
        return;
    }

    generateData(fout);
}

// Структура для хранения данных точки
struct Point {
    double x, y;
    std::vector<double> coordinates;
    int classId;
    std::vector<double> features;
    int label;
};

// Функция для вычисления метрики Евклида
double euclideanDistance(const Point& p1, const Point& p2) {
    double dist = 0.0;
    for (size_t i = 0; i < p1.coordinates.size(); ++i) {
        dist += std::pow(p1.coordinates[i] - p2.coordinates[i], 2);
    }
    return std::sqrt(dist);
}

std::vector<Point> readDataFromFile(const std::string& filename, int classes, int dimensions) {
    std::vector<Point> data;
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        return data;
    }
    std::string line;
    int currentClass = 1;
    int pointCount = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            currentClass++;
            pointCount = 0;
            continue;
        }
        if (pointCount >= classes) {
            continue;
        }
        std::stringstream ss(line);
        Point point;
        point.classId = currentClass;
        for (int i = 0; i < dimensions; ++i) {
            double coordinate;
            ss >> coordinate;
            point.coordinates.push_back(coordinate);
        }
        data.push_back(point);
        pointCount++;
    }
    return data;
}

// Функция для определения принадлежности кластеров к классам точек
std::vector<Point> clusterByClass(const std::vector<Point>& data, int k, std::function<std::vector<Point>(const std::vector<Point>&, int)> clusteringAlgorithm, const std::string& outputFilename) {
    std::vector<Point> allCentroids;
    std::vector<std::vector<Point>> dataByClass;
    int currentClass = 1;
    std::vector<Point> currentClassData;

    for (const auto& point : data) {
        if (point.classId == currentClass) {
            currentClassData.push_back(point);
        } else {
            dataByClass.push_back(currentClassData);
            currentClass++;
            currentClassData.clear();
            currentClassData.push_back(point);
        }
    }
    dataByClass.push_back(currentClassData);

    std::ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file: " << outputFilename << std::endl;
        return allCentroids;
    }
    outputFile << std::fixed << std::setprecision(10);

    for (size_t i = 0; i < dataByClass.size(); ++i) {
        if (!dataByClass[i].empty()) {
            std::vector<Point> centroids = clusteringAlgorithm(dataByClass[i], k);
            for (auto& centroid : centroids) {
                centroid.classId = i + 1;
                allCentroids.push_back(centroid);
                outputFile << centroid.x << " " << centroid.y << std::endl;
            }
        }
    }
    outputFile.close();
    return allCentroids;
}

// Алгоритмы кластеризации
// 1. Алгоритм простейшей расстановки центров кластеров
std::vector<Point> simpleClustering(const std::vector<Point>& data, int k) {
    std::vector<Point> centroids;
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0); // Заполняем вектор индексами от 0 до data.size()-1
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g); // Перемешиваем индексы случайным образом

    // Выбираем первые k точек из перемешанного набора индексов в качестве центроидов
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data[indices[i]]);
    }
    return centroids;
}

// 2. Алгоритм, основанный на методе просеивания
std::vector<Point> sieveMethod(const std::vector<Point>& data, int k) {
    std::vector<Point> centroids;
    if (data.empty() || k <= 0 || k > data.size()) return centroids;
    centroids.push_back(data[0]); // Первый центроид - первая точка данных

    // Итеративно выбираем центроиды
    for (int i = 1; i < k; ++i) {
        double maxMinDist = -1.0; // Максимальное минимальное расстояние (инициализируем малым значением)
        int bestIndex = -1; // Индекс точки, которая будет следующим центроидом
        
        // Перебираем все точки данных
        for (size_t j = 0; j < data.size(); ++j) {
            double minDist = std::numeric_limits<double>::max(); // Максимальное расстояние до уже выбранных центроидов (инициализируем большим значением)
            // Находим минимальное расстояние до уже выбранных центроидов
            for (const auto& centroid : centroids) {
                minDist = std::min(minDist, euclideanDistance(data[j], centroid));
            }
            // Если минимальное расстояние больше текущего максимума, обновляем максимум и индекс
            if (minDist > maxMinDist) {
                maxMinDist = minDist;
                bestIndex = j;
            }
        }
        centroids.push_back(data[bestIndex]); // Добавляем точку с максимальным минимальным расстоянием в качестве следующего центроида
    }
    return centroids;
}

// 3. Алгоритм максиминного расстояния
std::vector<Point> maxminDistance(const std::vector<Point>& data, int k) {
    std::vector<Point> centroids;
    if (data.empty() || k <= 0 || k > data.size()) return centroids;
    centroids.push_back(data[0]); // Первый центроид - первая точка данных
    
    // Итеративно выбираем центроиды
    for (int i = 1; i < k; ++i) {
        double maxDist = -1.0; // Максимальное минимальное расстояние до уже выбранных центроидов (инициализируем малым значением)
        int bestIndex = -1; // Индекс точки, которая будет следующим центроидом

        // Перебираем все точки данных
        for (size_t j = 0; j < data.size(); ++j) {
            double minDistToCentroids = std::numeric_limits<double>::max(); // Минимальное расстояние до уже выбранных центроидов (инициализируем большим значением)
            // Находим минимальное расстояние до уже выбранных центроидов
            for (const auto& centroid : centroids) {
                minDistToCentroids = std::min(minDistToCentroids, euclideanDistance(data[j], centroid));
            }
            // Если минимальное расстояние больше текущего максимума, обновляем максимум и индекс
            if (minDistToCentroids > maxDist) {
                maxDist = minDistToCentroids;
                bestIndex = j;
            }
        }
        centroids.push_back(data[bestIndex]); // Добавляем точку с максимальным минимальным расстоянием в качестве следующего центроида
    }
    return centroids;
}

// Алгоритм kmeans "уравнивает" центроиды, инициализированные прошлыми тремя алгоритмами
std::vector<Point> kMeans(const std::vector<Point>& data, int k, const std::vector<Point>& initialCentroids, int maxIterations = 1000) {
    std::vector<Point> centroids = initialCentroids;
    std::vector<int> assignments(data.size());
    int dimensions = data[0].coordinates.size(); // Получаем размерность данных из первой точки

    for (int iter = 0; iter < maxIterations; ++iter) {
        for (size_t i = 0; i < data.size(); ++i) {
            double minDist = std::numeric_limits<double>::max();
            for (size_t j = 0; j < centroids.size(); ++j) {
                double dist = euclideanDistance(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    assignments[i] = j;
                }
            }
        }
        std::vector<Point> newCentroids(k);
        for (int i = 0; i < k; ++i) {
            newCentroids[i].coordinates.resize(dimensions); // Инициализируем векторы координат
        }
        std::vector<int> counts(k, 0);
        for (size_t i = 0; i < data.size(); ++i) {
            int clusterIndex = assignments[i];
            counts[clusterIndex]++;
            for (size_t j = 0; j < data[i].coordinates.size(); ++j) {
                newCentroids[clusterIndex].coordinates[j] += data[i].coordinates[j];
            }
        }

        for (int i = 0; i < k; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < newCentroids[i].coordinates.size(); ++j) {
                    newCentroids[i].coordinates[j] /= counts[i];
                }
            } else {
                // Обработка пустого кластера: инициализация случайной точкой
                int randomIndex = rand() % data.size();
                newCentroids[i] = data[randomIndex];
            }
        }
        centroids = newCentroids;
    }
    return centroids;
}

// Функция для сохранения центроидов в файл
void saveCentroids(const std::vector<Point>& centroids, const std::string& filename) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error opening " << filename << std::endl;
        return;
    }
    for (const auto& centroid : centroids) {
        for (double coord : centroid.coordinates) {
            fout << coord << " ";
        }
        fout << std::endl;
    }
    fout.close();
}

// Класс с реализацией перцептрона для предугадывания принадлежности точек к классам
class Perceptron {
private:
    std::vector<double> weights;
    double bias;
    double learningRate;

    // Функция для нормализации данных
    void normalizeData(std::vector<Point>& data) {
        int numFeatures = data[0].coordinates.size(); // число признаков
        for (int i = 0; i < numFeatures; ++i) {
            double min_val = std::numeric_limits<double>::max();
            double max_val = std::numeric_limits<double>::min();
            for (auto& point : data) {
                min_val = std::min(min_val, point.coordinates[i]);
                max_val = std::max(max_val, point.coordinates[i]);
            }
            for (auto& point : data) {
                if (max_val - min_val > 0) {
                    point.coordinates[i] = (point.coordinates[i] - min_val) / (max_val - min_val);
                }
                else {
                    point.coordinates[i] = 0.0; // Обработка случая, когда все значения одинаковые
                }
            }
        }
    }

public:
    Perceptron(int numFeatures, double learningRate) : learningRate(learningRate) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 0.5); // Изменена инициализация весов

        weights.resize(numFeatures);
        for (auto& w : weights) {
            w = dist(gen);
        }
        bias = 0;
    }

    int predict(const std::vector<double>& features) const {
        double sum = bias;
        for (size_t i = 0; i < features.size(); ++i) {
            sum += weights[i] * features[i];
        }
        // std::cout << sum << " ee" << std::endl;
        return (sum <= 0) ? 1 : 2;
    }

    void train(std::vector<Point>& data, int epochs) {
        normalizeData(data); // Нормализуем данные перед обучением

        for (int epoch = 0; epoch < epochs; ++epoch) {
            int correct_classifications = 0;
            for (auto& point : data) {
                point.features = point.coordinates;
                int prediction = predict(point.features);
                // std::cout << prediction << " ew33ee" << std::endl;
                int error = point.label - prediction;
                bias += error * learningRate;
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] += error * learningRate * point.features[i];
                }
                if (prediction == point.label) {
                    correct_classifications++;
                }
            }
            std::cout << "Epoch " << epoch + 1 << ": Correct classifications = " << correct_classifications << "/" << data.size() << std::endl;
            //Вывод весов и смещения для отладки после каждой эпохи
            std::cout << "Weights: ";
            for (double w : weights) std::cout << w << " ";
            std::cout << std::endl;
            std::cout << "Bias: " << bias << std::endl;

        }
    }

    void saveWeights(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << bias << std::endl;
            for (double w : weights) {
                file << w << " ";
            }
            file << std::endl;
            file.close();
        }
        else {
            std::cerr << "Error opening file: " << filename << std::endl;
        }
    }

    void classifyAndSave(const std::vector<Point>& data, const std::string& outputFilename) const {
        std::ofstream outputFile(outputFilename);
        if (!outputFile.is_open()) {
            std::cerr << "Error opening file: " << outputFilename << std::endl;
            return;
        }
        for (const auto& point : data) {
            outputFile << point.label << " " << predict(point.features) << std::endl;
        }
        outputFile.close();
    }
};

int main() {
    srand(time(0));

    DataGenerator generator("config.txt"); //Если вам нужна генерация
    generator.generateAndSave("new_data.txt");

    std::vector<Point> data = readDataFromFile("new_data.txt", 2, 2);

    for (auto& point : data) {
        point.label = point.classId;
    }

    int k = 2; // Количество кластеров
    int maxIterations = 1000; // Увеличенное количество итераций

    auto runKMeansAndSave = [&](const std::string& initMethod, const std::string& filename) {
        std::vector<Point> initialCentroids;
        if (initMethod == "simpleClustering") initialCentroids = simpleClustering(data, k);
        else if (initMethod == "sieveMethod") initialCentroids = sieveMethod(data, k);
        else if (initMethod == "maxminDistance") initialCentroids = maxminDistance(data, k);
        else {
            std::cerr << "Unknown initialization method: " << initMethod << std::endl;
            return;
        }

        std::vector<Point> finalCentroids = kMeans(data, k, initialCentroids, maxIterations);
        saveCentroids(finalCentroids, filename);
        std::cout << initMethod << " centroids saved to " << filename << std::endl;
        };

    runKMeansAndSave("simpleClustering", "simple_centroids.txt");
    runKMeansAndSave("sieveMethod", "sieve_centroids.txt");
    runKMeansAndSave("maxminDistance", "maxmin_centroids.txt");

    Perceptron perceptron(2, 0.001); 
    perceptron.train(data, 100);
    perceptron.saveWeights("perceptron_weights.txt");
    perceptron.classifyAndSave(data, "perceptron_classification.txt");

    return 0;
}