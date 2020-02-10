//
//  RectTest.swift
//
//  Created by Giles Payne on 2020/01/31.
//

import XCTest
import StitchApp

class RectTest: OpenCVTestCase {

    let r = Rect2i()
    let rect = Rect2i(x: 0, y: 0, width: 10, height: 10)

    func testArea() {
        let area = rect.area()
        XCTAssertEqual(100.0, area)
    }

    func testBr() {
        let p_br = rect.br()
        let truth = Point2i(x: 10, y: 10)
        XCTAssertEqual(truth, p_br)
    }

    func testClone() {
        let r = rect.clone()
        XCTAssertEqual(rect, r)
    }

    func testContains() {
        let rect = Rect2i(x: 0, y: 0, width: 10, height: 10)

        let p_inner = Point2i(x: 5, y: 5)
        let p_outer = Point2i(x: 5, y: 55)
        let p_bl = Point2i(x: 0, y: 0)
        let p_br = Point2i(x: 10, y: 0)
        let p_tl = Point2i(x: 0, y: 10)
        let p_tr = Point2i(x: 10, y: 10)

        XCTAssert(rect.contains(p_inner))
        XCTAssert(rect.contains(p_bl))

        XCTAssertFalse(rect.contains(p_outer))
        XCTAssertFalse(rect.contains(p_br))
        XCTAssertFalse(rect.contains(p_tl))
        XCTAssertFalse(rect.contains(p_tr))
    }

    func testEqualsObject() {
        var flag = rect == r
        XCTAssertFalse(flag)

        let r = rect.clone()
        flag = rect == r
        XCTAssert(flag)
    }

    func testHashCode() {
        XCTAssertEqual(rect.hash(), rect.hash())
    }

    func testRect() {
        let r = Rect2i()

        XCTAssertEqual(0, r.x)
        XCTAssertEqual(0, r.y)
        XCTAssertEqual(0, r.width)
        XCTAssertEqual(0, r.height)
    }

    func testRectDoubleArray() {
        let vals:[Double] = [1, 3, 5, 2]
        let r = Rect2i(vals: vals as [NSNumber])
        
        XCTAssertEqual(1, r.x)
        XCTAssertEqual(3, r.y)
        XCTAssertEqual(5, r.width)
        XCTAssertEqual(2, r.height)
    }

    func testRectIntIntIntInt() {
        let rect = Rect2i(x: 1, y: 3, width: 5, height: 2)

        XCTAssertNotNil(rect)
        XCTAssertEqual(1, rect.x)
        XCTAssertEqual(3, rect.y)
        XCTAssertEqual(5, rect.width)
        XCTAssertEqual(2, rect.height)
    }

    func testRectPointPoint() {
        let p1 = Point2i(x:4, y:4)
        let p2 = Point2i(x: 2, y: 3)

        let r = Rect2i(point: p1, point: p2)
        XCTAssertNotNil(r);
        XCTAssertEqual(2, r.x);
        XCTAssertEqual(3, r.y);
        XCTAssertEqual(2, r.width);
        XCTAssertEqual(1, r.height);
    }

    func testRectPointSize() {
        let p1 = Point2i(x: 4, y: 4)
        let sz = Size2i(width: 3, height: 1)
        let r = Rect2i(point: p1, size: sz)

        XCTAssertEqual(4, r.x)
        XCTAssertEqual(4, r.y)
        XCTAssertEqual(3, r.width)
        XCTAssertEqual(1, r.height)
    }

    func testSet() {
        let vals1:[Double] = []
        let r1 = Rect2i(vals:vals1 as [NSNumber])

        XCTAssertEqual(0, r1.x)
        XCTAssertEqual(0, r1.y)
        XCTAssertEqual(0, r1.width)
        XCTAssertEqual(0, r1.height)

        let vals2:[Double] = [2, 2, 10, 5]
        let r = Rect2i(vals: vals2 as [NSNumber])

        XCTAssertEqual(2, r.x)
        XCTAssertEqual(2, r.y)
        XCTAssertEqual(10, r.width)
        XCTAssertEqual(5, r.height)
    }

    func testSize() {
        let s1 = Size2i(width: 0, height: 0)
        XCTAssertEqual(s1, r.size())

        let s2 = Size2i(width: 10, height: 10)
        XCTAssertEqual(s2, rect.size())
    }

    func testTl() {
        let p_tl = rect.tl()
        let truth = Point2i(x: 0, y: 0)
        XCTAssertEqual(truth, p_tl)
    }

    func testToString() {
        let actual = "\(rect)"
        let expected = "Rect2i {0,0,10,10}"
        XCTAssertEqual(expected, actual);
    }

}
