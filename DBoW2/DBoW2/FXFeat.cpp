#include <sstream>
#include "FXFeat.h"

using namespace std;

namespace DBoW2 {

const int FXFeat::L=256;


void FXFeat::meanValue(const std::vector<FXFeat::pDescriptor> &descriptors,
                       FXFeat::TDescriptor &mean) {
    if (descriptors.empty()) {
        mean.release();
        return;
    } else if (descriptors.size() == 1) {
        mean = descriptors[0]->clone();
    } else {
        mean.create(1, descriptors[0]->cols, descriptors[0]->type());
        mean.setTo(0);
        for (auto descriptor : descriptors) {
            mean += *descriptor;
        }
        mean *= (1.0 / (double)descriptors.size());
    }
}


int FXFeat::distance(const FXFeat::TDescriptor &a, const FXFeat::TDescriptor &b) {
    return 0;
}


}
