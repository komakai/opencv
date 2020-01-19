//
//  MatOfPoint2f.mm
//
//  Created by Giles Payne on 2019/12/27.
//

#import "MatOfPoint2f.h"
#import "Range.h"
#import "CVPoint.h"
#import "CVType.h"

@implementation MatOfPoint2f

const int _depth = CV_32F;
const int _channels = 2;

#ifdef __cplusplus
- (instancetype)initWithNativeMat:(cv::Mat*)nativeMat {
    self = [super initWithNativeMat:nativeMat];
    if (self && ![self empty] && [self checkVector:_channels depth:_depth] < 0) {
        @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Incompatible Mat" userInfo:nil];
    }
    return self;
}
#endif

- (instancetype)initWithMat:(Mat*)mat {
    self = [super initWithMat:mat rowRange:[Range all]];
    if (self && ![self empty] && [self checkVector:_channels depth:_depth] < 0) {
        @throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Incompatible Mat" userInfo:nil];
    }
    return self;
}

- (instancetype)initWithArray:(NSArray<CVPoint*>*)array {
    self = [super init];
    if (self) {
        [self fromArray:array];
    }
    return self;
}

- (void)alloc:(int)elemNumber {
    if (elemNumber>0) {
        [super create:elemNumber cols:1 type:[CVType makeType:_depth channels:_channels]];
    }
}

- (void)fromArray:(NSArray<CVPoint*>*)array {
    NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:array.count * _channels];
    for (int index = 0; index < array.count; index++) {
        data[_channels * index] = [NSNumber numberWithFloat:array[index].x];
        data[_channels * index + 1] = [NSNumber numberWithFloat:array[index].y];
    }
    [self alloc:(int)array.count];
    [self put:0 col:0 data:data];
}

- (NSArray<CVPoint*>*)toArray {
    int length = [self length] / _channels;
    NSMutableArray<CVPoint*>* ret = [[NSMutableArray alloc] initWithCapacity:length];
    if (length > 0) {
        NSMutableArray<NSNumber*>* data = [[NSMutableArray alloc] initWithCapacity:length];
        [self get:0 col:0 data:data];
        for (int index = 0; index < length; index++) {
            ret[index] = [[CVPoint alloc] initWithX:data[index * _channels].floatValue y:data[index * _channels + 1].floatValue];
        }
    }
    return ret;
}

- (int)length {
    int num = [self checkVector:_channels depth:_depth];
    if (num < 0) {
        @throw  [NSException exceptionWithName:NSInternalInconsistencyException reason:@"Incompatible Mat" userInfo:nil];
    }
    return num * _channels;
}

@end
