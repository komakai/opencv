//
//  Point2i.h
//
//  Created by Giles Payne on 2019/10/09.
//

#pragma once

#ifdef __cplusplus
#import "opencv.hpp"
#endif

#import <Foundation/Foundation.h>

@class Rect2i;

NS_ASSUME_NONNULL_BEGIN

NS_SWIFT_NAME(Point)
@interface Point2i : NSObject

@property int x;
@property int y;
#ifdef __cplusplus
@property(readonly) cv::Point2i& nativeRef;
#endif

- (instancetype)init;
- (instancetype)initWithX:(int)x y:(int)y;
- (instancetype)initWithVals:(NSArray<NSNumber*>*)vals;

#ifdef __cplusplus
+ (instancetype)fromNative:(cv::Point2i&)point;
- (void)update:(cv::Point2i&)point;
#endif
- (Point2i*)clone;
- (double)dot:(Point2i*)point;
- (BOOL)inside:(Rect2i*)rect;

- (void)set:(NSArray<NSNumber*>*)vals NS_SWIFT_NAME(set(vals:));
- (BOOL)isEqual:(nullable id)other;
- (NSUInteger)hash;
- (NSString *)description;
@end

NS_ASSUME_NONNULL_END
