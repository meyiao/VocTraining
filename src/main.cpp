//
// Created by tan on 18-6-1.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ORBVocabulary.h"
#include "utils.h"
#include "ORBextractor.h"

#define DEBUG_SHOW 1

using namespace cv;
using namespace std;
using namespace DBoW2;

const int K = 10;
const int L = 5;

int main() {

    // load images
    vector<string> images;
    Utils::LoadImages("../train_data/images.txt", images);

    // create features
    cout << "creating features..." << endl;
    ORB_SLAM2::ORBextractor orb(600, 1.2, 3, 20, 10);
    vector<vector<Mat> > features;
    for (size_t i=0; i<images.size(); i+=1) {
        cout << i << endl;
        Mat im = imread("../train_data/"+images[i], CV_LOAD_IMAGE_GRAYSCALE);
        vector<KeyPoint> keys;
        Mat desc;
        orb(im, Mat(), keys, desc);

        if (keys.size()<200) {
            cout << "oh no!" << endl;
            continue;
        }

        features.push_back(vector<cv::Mat>());
        Utils::ChangeStructure(desc, features.back());

#if DEBUG_SHOW
        cv::Mat cimg;
        drawKeypoints(im, keys, cimg, Scalar(0,255,0));
        imshow("feature", cimg);
        waitKey(10);
#endif

    }
    cout << "... done!\n\n";


    // creating vocabulary
    cout << "Creating a " << K << "^" << L << " vocabulary...";

    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    ORBVocabulary voc(K, L, weight, score);
    voc.create(features);
    cout << "... done!\n\n";

    cout << "Vocabulary information: " << endl << voc << "\n\n";

    // save the vocabulary to disk
    cout << "Saving vocabulary...";
    voc.saveToBinaryFile("../vocSmall.bin");
    cout << "... done!\n\n";

    return 0;
}
