#include <iostream>
#include <opencv2/opencv.hpp>
#include "ORBextractor.h"
#include "ORBVocabulary.h"
#include "utils.h"


struct KeyFrame {

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat img;
    DBoW2::BowVector bowVec;

};

using KeyFramePtr = std::shared_ptr<KeyFrame>;

int main() {

    const std::string databaseDir = "D:/work/data/evo/poster/keyframes";
    const std::string queryDir = "D:/work/data/evo/poster3/keyframes";

    // database images
    std::vector<std::string> databaseImgFiles;
    cv::glob(databaseDir, databaseImgFiles);
    std::cout << "found (" << databaseImgFiles.size() << ") database images!\n";

    // query images
    std::vector<std::string> queryImgFiles;
    cv::glob(queryDir, queryImgFiles);
    std::cout << "found (" << queryImgFiles.size() << ") query images!\n";

    // ORB feature extractor
    ORB_SLAM2::ORBextractor orb(600, 1.2, 3, 20, 10);

    // Vocabulary
    ORBVocabulary voc(10, 5);
    voc.loadFromBinaryFile("../vocSmall.bin");
    std::cout << "vocab loaded!\n";

    std::vector<KeyFramePtr> database;
    for (int i = 0; i < databaseImgFiles.size(); ++i) {
        cv::Mat img = cv::imread(databaseImgFiles[i], cv::IMREAD_GRAYSCALE);
        img = img(cv::Rect(100, 100, 520, 520));
        cv::resize(img, img, cv::Size(640, 640));

        KeyFramePtr keyframe = std::make_shared<KeyFrame>();
        keyframe->img = img.clone();
        // extract features
        orb(img, cv::Mat(), keyframe->keypoints, keyframe->descriptors);

        // compute BoW
        std::vector<cv::Mat> descVec;
        Utils::ChangeStructure(keyframe->descriptors, descVec);
        voc.transform(descVec, keyframe->bowVec);

        // draw keypoints
        cv::Mat cimg;
        cv::drawKeypoints(img, keyframe->keypoints, cimg, cv::Scalar::all(-1));
        cv::imshow("frame", cimg);
        cv::waitKey(2);

        database.push_back(keyframe);
    }

    // query
    for (int i = 0; i < queryImgFiles.size(); ++i) {
        cv::Mat img = cv::imread(queryImgFiles[i], cv::IMREAD_GRAYSCALE);
        img = img(cv::Rect(100, 100, 520, 520));
        cv::resize(img, img, cv::Size(640, 640));

        KeyFramePtr keyframe = std::make_shared<KeyFrame>();
        keyframe->img = img.clone();
        // extract features
        orb(img, cv::Mat(), keyframe->keypoints, keyframe->descriptors);

        // compute BoW
        std::vector<cv::Mat> descVec;
        Utils::ChangeStructure(keyframe->descriptors, descVec);
        keyframe->bowVec.clear();
        voc.transform(descVec, keyframe->bowVec);

        // find the best 5 candidates from database
        std::vector<std::pair<int, double>> scores;
        for (int j = 0; j < database.size(); ++j) {
            double score = voc.score(database[j]->bowVec, keyframe->bowVec);
            scores.emplace_back(j, score);
        }

        std::sort(scores.begin(), scores.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
            return a.second > b.second;
        });

        // show the best match
        cv::Mat cimg;
        cv::hconcat(database[scores[0].first]->img, keyframe->img, cimg);
        cv::putText(cimg, "score: " + std::to_string(scores[0].second),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("best match", cimg);
        cv::waitKey();


    }

    return 0;

}