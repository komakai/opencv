//
//  CascadeClassifierTest.swift
//
//  Created by Giles Payne on 2021/05/10.
//

import XCTest
import OpenCV

class CascadeClassifierTest: XCTestCase {

    private CascadeClassifier cc;

    @Override
    protected void setUp() throws Exception {
        super.setUp();

        cc = null;
    }

    func testCascadeClassifier() {
        cc = new CascadeClassifier();
        assertNotNull(cc);
    }

    func testCascadeClassifierString() {
        cc = new CascadeClassifier(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        assertNotNull(cc);
    }

    func testDetectMultiScaleMatListOfRect() {
        CascadeClassifier cc = new CascadeClassifier(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        MatOfRect faces = new MatOfRect();

        Mat greyLena = new Mat();
        Imgproc.cvtColor(rgbLena, greyLena, Imgproc.COLOR_RGB2GRAY);
        Imgproc.equalizeHist(greyLena, greyLena);

        cc.detectMultiScale(greyLena, faces, 1.1, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
        assertEquals(1, faces.total());
    }

    func testEmpty() {
        cc = new CascadeClassifier();
        assertTrue(cc.empty());
    }

    func testLoad() {
        cc = new CascadeClassifier();
        cc.load(OpenCVTestRunner.LBPCASCADE_FRONTALFACE_PATH);
        assertFalse(cc.empty());
    }

}
