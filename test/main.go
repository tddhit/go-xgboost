package main

import (
	"bufio"
	"os"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	xgboost "github.com/tddhit/go-xgboost"
	"github.com/tddhit/tools/log"
)

var (
	featurePool sync.Pool
	resultPool  sync.Pool
)

func init() {
	featurePool.New = func() interface{} {
		return make([]float32, 46)
	}
	resultPool.New = func() interface{} {
		return make([]float32, 1)
	}
}

func main() {
	booster, err := xgboost.XGBoosterCreate()
	if err != nil {
		log.Fatal(err)
	}
	err = xgboost.XGBoosterLoadModel(booster, "0004.model")
	if err != nil {
		log.Fatal(err)
	}
	file, _ := os.Open("mq2008.test")
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		features := strings.Split(line, " ")[1:]
		featureVec := featurePool.Get().([]float32)
		for i, _ := range featureVec {
			featureVec[i] = 0.0
		}
		for i, _ := range features {
			s := strings.Split(features[i], ":")
			id, _ := strconv.Atoi(s[0])
			value, _ := strconv.ParseFloat(s[1], 32)
			featureVec[id-1] = float32(value)
		}
		dmat, err := xgboost.XGDMatrixCreateFromMat(
			unsafe.Pointer(&featureVec[0]), 1, 46, 0.0)
		if err != nil {
			log.Fatal(err)
		}
		result := resultPool.Get().([]float32)
		err = xgboost.XGBoosterPredict(booster, dmat, 0, 0, result)
		if err != nil {
			log.Fatal(err)
		}
		log.Info(result)
		xgboost.XGDMatrixFree(dmat)
		resultPool.Put(result)
		featurePool.Put(featureVec)
	}
	xgboost.XGBoosterFree(booster)
}
