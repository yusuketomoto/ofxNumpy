#pragma once

#include "cnpy.h"

#define OFX_NUMPY_BEGIN_NAMESPACE namespace ofx { namespace Numpy {
#define OFX_NUMPY_END_NAMESPACE } }

OFX_NUMPY_BEGIN_NAMESPACE

using namespace cnpy;

namespace detail {
    template <int N>
    struct convert_impl {
        template <typename T1, typename T2>
        static void convert(T1* o1, size_t n, vector<T2>& o2);
    };
    template<>
    struct convert_impl<1> {
        template <typename T1, typename T2>
        static void convert(T1* o1, size_t n, vector<T2>& o2) {
            for (int i=0; i<n; i++) {
                T2 c(o1[i]);
                o2.push_back(c);
            }
        }
    };
    template<>
    struct convert_impl<2> {
        template <typename T1, typename T2>
        static void convert(T1* o1, size_t n, vector<T2>& o2) {
            for (int i=0; i<n; i+=2) {
                T2 c(o1[i], o1[i+1]);
                o2.push_back(c);
            }
        }
    };
    template<>
    struct convert_impl<3> {
        template <typename T1, typename T2>
        static void convert(T1* o1, size_t n, vector<T2>& o2) {
            for (int i=0; i<n; i+=3) {
                T2 c(o1[i], o1[i+1], o1[i+2]);
                o2.push_back(c);
            }
        }
    };
    template<>
    struct convert_impl<4> {
        template <typename T1, typename T2>
        static void convert(T1* o1, size_t n, vector<T2>& o2) {
            for (int i=0; i<n; i+=4) {
                T2 c(o1[i], o1[i+1], o1[i+2], o1[i+3]);
                o2.push_back(c);
            }
        }
    };
}
template <int N, class T1, class T2>
void convert(T1* o1, size_t n, vector<T2>& o2)
{
    detail::convert_impl<N>::convert(o1, n, o2);
}

template <typename T1, typename T2>
void toOf(const T1& o1, vector<T2>& o2);

#define CONV(o, N, DIM)                         \
if (o.data_type == 'f') {                       \
    if (o.word_size == sizeof(float))           \
        convert<DIM>(o.data<float>(), n, o2);   \
    else if (o.word_size == sizeof(double))     \
        convert<DIM>(o.data<double>(), n, o2);  \
}                                               \
else if (o.data_type == 'i') {                  \
    if (o.word_size == sizeof(int))             \
        convert<DIM>(o.data<int>(), n, o2);     \
    else if (o.word_size == sizeof(int64_t))    \
        convert<DIM>(o.data<int64_t>(), n, o2); \
}                                               \
else if (o.data_type == 'u') {                  \
    if (o.word_size == sizeof(uint))            \
        convert<DIM>(o.data<uint>(), n, o2);    \
    else if (o.word_size == sizeof(uint64_t))   \
        convert<DIM>(o.data<uint64_t>(), n, o2);\
}

void getSize(const NpyArray& o, size_t& dim, size_t& n)
{
    if (o.shape.size() == 2) {
        dim = o.shape[1];
        n = o.shape[0] * o.shape[1];
    }
    else if (o.shape.size() == 3) {
        dim = o.shape[2];
        n = o.shape[0] * o.shape[1] * o.shape[2];
    }
}

template <>
void toOf(const NpyArray& o1, vector<ofFloatColor>& o2)
{
    o2.clear();
    size_t dim, n;
    getSize(o1, dim, n);

    assert(dim > 0 && dim < 5);
    
    switch (dim) {
        case 1: CONV(o1, n, 1) break;
        case 2: CONV(o1, n, 2) break;
        case 3: CONV(o1, n, 3) break;
        case 4: CONV(o1, n, 4) break;
        default: break;
    }
}

template <>
void toOf(const NpyArray& o1, vector<ofVec2f>& o2)
{
    o2.clear();
    size_t dim, n;
    getSize(o1, dim, n);
    assert(dim > 0 && dim < 3);
    
    switch (dim) {
        case 1: CONV(o1, n, 1) break;
        case 2: CONV(o1, n, 2) break;
        default: break;
    }
}

template <>
void toOf(const NpyArray& o1, vector<ofVec3f>& o2)
{
    o2.clear();
    size_t dim, n;
    getSize(o1, dim, n);
    assert(dim > 0 && dim < 4);
    
    switch (dim) {
        case 1: CONV(o1, n, 1) break;
        case 2: CONV(o1, n, 2) break;
        case 3: CONV(o1, n, 3) break;
        default: break;
    }
}

template <>
void toOf(const NpyArray& o1, vector<ofVec4f>& o2)
{
    o2.clear();
    size_t dim, n;
    getSize(o1, dim, n);
    assert(dim == 1 || dim == 4);
    
    switch (dim) {
        case 1: CONV(o1, n, 1) break;
        case 4: CONV(o1, n, 4) break;
        default: break;
    }
}

template <>
void toOf(const NpyArray& o1, vector<ofQuaternion>& o2)
{
    o2.clear();
    size_t dim, n;
    getSize(o1, dim, n);
    assert(dim == 4);
    
    switch (dim) {
        case 4: CONV(o1, n, 4) break;
        default: break;
    }
}

template <typename T>
bool load(const string& path, T& data)
{
    if (!ofFile::doesFileExist(path)) {
        ofLogError("couldn't load file") << path;
        return false;
    }
    
    NpyArray array = npy_load(ofToDataPath(path));
    toOf(array, data);
    return true;
}

#undef CONV

OFX_NUMPY_END_NAMESPACE

namespace ofxNumpy = ofx::Numpy;