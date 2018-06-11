//
// Created by tan on 18-4-28.
//

#ifndef INIALIGN_UTILS_H
#define INIALIGN_UTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>


class Utils {

public:

    Utils();

    static void LoadImages(const std::string& file,
                           std::vector<std::string>& imgNames)
    {
        std::fstream fs(file);
        if (!fs.is_open()) {
            std::cerr << "Error: failed to open file: " << file << "\n";
            exit(-2);
        }

        std::string str;
        while (!fs.eof()) {
            if (getline(fs, str)) {
                std::stringstream ss(str);
                imgNames.push_back(str);
            }
        }
        std::cout << "Finish reading images: " << imgNames.size() << " images in total!\n";
    }


    static void ChangeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out) {
        out.resize(plain.rows);
        for (int i=0; i<plain.rows; ++i) {
            out[i] = plain.row(i);
        }
    }


};


#endif //INIALIGN_UTILS_H
