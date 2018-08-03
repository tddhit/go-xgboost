package xgboost

// #cgo LDFLAGS: -L${SRCDIR}/lib -lxgboost -ldmlc -lrabit -lpthread -lm -lrt
// #cgo CFLAGS: -I${SRCDIR}/include
// #include <stdlib.h>
// #include "xgboost/c_api.h"
import "C"
import (
	"errors"
	"unsafe"
)

var (
	errCreateBooster       = errors.New("create booster failed.")
	errLoadModel           = errors.New("load model failed.")
	errCreateDMatrixHandle = errors.New("create dmatrix failed")
	errPredict             = errors.New("predict failed.")
	errFreeDMatrix         = errors.New("free dmatrix failed.")
	errFreeBooster         = errors.New("free booster failed.")
)

type DMatrixHandle struct {
	pointer C.DMatrixHandle
}

type BoosterHandle struct {
	pointer C.BoosterHandle
}

type DMatrix [][]float32

/*!
 * \brief create xgboost learner
 * \param dmats matrices that are set to be cached
 * \param len length of dmats
 * \param out handle to the result booster
 * \return 0 when success, -1 when failure happens

int XGBoosterCreate(const DMatrixHandle dmats[],
                    bst_ulong len,
                    BoosterHandle *out);
*/

func XGBoosterCreate() (*BoosterHandle, error) {
	var handle C.BoosterHandle
	ret := C.XGBoosterCreate(
		nil,
		0,
		&handle,
	)
	if C.int(ret) == -1 {
		return nil, errCreateBooster
	}
	return &BoosterHandle{handle}, nil
}

/*!
 * \brief load model from existing file
 * \param handle handle
 * \param fname file name
* \return 0 when success, -1 when failure happens

int XGBoosterLoadModel(BoosterHandle handle,
                       const char *fname);
*/

func XGBoosterLoadModel(handle *BoosterHandle, fname string) error {
	ret := C.XGBoosterLoadModel(
		handle.pointer,
		C.CString(fname),
	)
	if C.int(ret) == -1 {
		return errLoadModel
	}
	return nil
}

/*!
 * \brief create matrix content from dense matrix
 * \param data pointer to the data space
 * \param nrow number of rows
 * \param ncol number columns
 * \param missing which value to represent missing value
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens

int XGDMatrixCreateFromMat(const float *data,
                           bst_ulong nrow,
                           bst_ulong ncol,
                           float missing,
                           DMatrixHandle *out);
*/

func XGDMatrixCreateFromMat(data unsafe.Pointer, nrow, ncol int,
	missing float32) (*DMatrixHandle, error) {

	var handle C.DMatrixHandle
	ret := C.XGDMatrixCreateFromMat(
		(*C.float)(data),
		C.bst_ulong(nrow),
		C.bst_ulong(ncol),
		C.float(missing),
		&handle,
	)
	if C.int(ret) == -1 {
		return nil, errCreateDMatrixHandle
	}
	return &DMatrixHandle{handle}, nil
}

/*!
 * \brief make prediction based on dmat
 * \param handle handle
 * \param dmat data matrix
 * \param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 *          4:output feature contributions to individual predictions
 * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * \param out_len used to store length of returning result
 * \param out_result used to set a pointer to array
 * \return 0 when success, -1 when failure happens

int XGBoosterPredict(BoosterHandle handle,
                     DMatrixHandle dmat,
                     int option_mask,
                     unsigned ntree_limit,
                     bst_ulong *out_len,
                     const float **out_result);
*/

func XGBoosterPredict(handle *BoosterHandle, dmat *DMatrixHandle,
	optionMask int, ntreeLimit uint, result []float32) error {

	var (
		outLen    C.bst_ulong
		outResult *C.float
	)
	ret := C.XGBoosterPredict(
		handle.pointer,
		dmat.pointer,
		C.int(optionMask),
		C.unsigned(ntreeLimit),
		&outLen,
		&outResult,
	)
	if C.int(ret) == -1 {
		return errPredict
	}
	length := int(outLen)
	ptr := outResult
	size := uintptr(unsafe.Sizeof(*ptr))
	for i := 0; i < length; i++ {
		result[i] = float32(*ptr)
		ptr = (*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + size))
	}
	return nil
}

/*!
 * \brief free space in data matrix
 * \return 0 when success, -1 when failure happens

int XGDMatrixFree(DMatrixHandle handle);
*/

func XGDMatrixFree(handle *DMatrixHandle) error {
	ret := C.XGDMatrixFree(handle.pointer)
	if C.int(ret) == -1 {
		return errFreeDMatrix
	}
	return nil
}

/*!
 * \brief free obj in handle
 * \param handle handle to be freed
 * \return 0 when success, -1 when failure happens

int XGBoosterFree(BoosterHandle handle);
*/

func XGBoosterFree(handle *BoosterHandle) error {
	ret := C.XGBoosterFree(handle.pointer)
	if C.int(ret) == -1 {
		return errFreeBooster
	}
	return nil
}
