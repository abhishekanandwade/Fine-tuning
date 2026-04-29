package handler

import (
	"net/http"
	"pmdm/core/domain"
	"time"

	"github.com/gin-gonic/gin"
)

// response represents a response body format
type Response struct {
	Success bool   `json:"success" example:"true"`
	Message string `json:"message" example:"Success"`
	Data    any    `json:"data,omitempty"`
}

// // newResponse is a helper function to create a response body
// func newResponse(success bool, message string, data any) Response {
// 	return Response{
// 		Success: success,
// 		Message: message,
// 		Data:    data,
// 	}
// }

// errorResponse represents an error response body format
// type errorResponse struct {
// 	Success bool   `json:"success" example:"false"`
// 	Message string `json:"message" example:"Error message"`
// }

// newErrorResponse is a helper function to create an error response body
// func newErrorResponse(message string) errorResponse {
// 	return errorResponse{
// 		Success: false,
// 		Message: message,
// 	}
// }

// meta represents metadata for a paginated response
// type meta struct {
// 	Total uint64 `json:"total" example:"100"`
// 	Limit uint64 `json:"limit" example:"10"`
// 	Skip  uint64 `json:"skip" example:"0"`
// }

// newMeta is a helper function to create metadata for a paginated response
// func newMeta(total, limit, skip uint64) meta {
// 	return meta{
// 		Total: total,
// 		Limit: limit,
// 		Skip:  skip,
// 	}
// }

// errorStatusMap is a map of defined error messages and their corresponding http status codes
// var errorStatusMap = map[error]int{
// 	port.ErrDataNotFound:               http.StatusNotFound,
// 	port.ErrConflictingData:            http.StatusConflict,
// 	port.ErrInvalidCredentials:         http.StatusUnauthorized,
// 	port.ErrUnauthorized:               http.StatusUnauthorized,
// 	port.ErrEmptyAuthorizationHeader:   http.StatusUnauthorized,
// 	port.ErrInvalidAuthorizationHeader: http.StatusUnauthorized,
// 	port.ErrInvalidAuthorizationType:   http.StatusUnauthorized,
// 	port.ErrInvalidToken:               http.StatusUnauthorized,
// 	port.ErrExpiredToken:               http.StatusUnauthorized,
// 	port.ErrForbidden:                  http.StatusForbidden,
// 	port.ErrNoUpdatedData:              http.StatusBadRequest,
// 	port.ErrInsufficientStock:          http.StatusBadRequest,
// 	port.ErrInsufficientPayment:        http.StatusBadRequest,
// }

// validationError sends an error response for some specific request validation error
func validationError(ctx *gin.Context, err error) {
	ctx.JSON(http.StatusBadRequest, err)
}

// handleError determines the status code of an error and returns a JSON response with the error message and status code
// func handleError(ctx *gin.Context, err error) {
// 	statusCode, ok := errorStatusMap[err]
// 	if !ok {
// 		statusCode = http.StatusInternalServerError
// 	}

// 	errRsp := newErrorResponse(err.Error())

// 	ctx.JSON(statusCode, errRsp)
// }

// handleAbort sends an error response and aborts the request with the specified status code and error message
// func handleAbort(ctx *gin.Context, err error) {
// 	statusCode, ok := errorStatusMap[err]
// 	if !ok {
// 		statusCode = http.StatusInternalServerError
// 	}

// 	rsp := newErrorResponse(err.Error())
// 	ctx.AbortWithStatusJSON(statusCode, rsp)
// }

// handleSuccess sends a success response with the specified status code and optional data
func handleSuccess(ctx *gin.Context, data any) {
	//rsp := newResponse(true, "Success", data)
	ctx.JSON(http.StatusOK, data)
}

func handleCreateSuccess(ctx *gin.Context, data any) {
	ctx.JSON(http.StatusCreated, data)
}

type CadreMasterResponse struct {
	CadreId   int    `json:"cadre_id"`
	CadreName string `json:"cadre_name"`
	GroupName string `json:"group_name"`
}

// func NewCadreMasterResponse(cadre *domain.CadreMaster) CadreMasterResponse {
// 	return CadreMasterResponse{
// 		CadreId:   cadre.CadreID,
// 		CadreName: cadre.CadreName,
// 		GroupName: cadre.GroupName,
// 	}
// }

// PostManagementResponse represents the response structure for PostManagementMaster
type PostManagementResponse struct {
	PostManagementID int64  `json:"postmanagement_id"`
	OfficeID         int64  `json:"office_id"`
	PostID           int64  `json:"post_id"`
	PostName         string `json:"post_name"`
	OfficeName       string `json:"office_name"`
	Status           string `json:"status"`
}

// NewPostManagementMasterResponse creates a new PostManagementMasterResponse instance
func NewPostManagementResponse(post domain.PostManagementMaster2) PostManagementResponse {
	return PostManagementResponse{
		PostManagementID: post.PostManagementID,
		OfficeID:         post.OfficeID,
		PostID:           post.PostID,
		PostName:         post.PostName,
		OfficeName:       post.OfficeName,
		Status:           post.Status,
	}
}

// // PostManagementFetchResponse represents the response structure for PostManagementMaster
// type PostManagementFetchResponse struct {
// 	OfficeID        int    `json:"office_id"`
// 	OfficeName      string `json:"office_name"`
// 	PostID          int    `json:"post_id"`
// 	PostName        string `json:"post_name"`
// 	GroupId         int    `json:"group_id"`
// 	Designation     string `json:"designation"`
// 	FilledStatus    string `json:"filled_status"`
// 	PermanentStatus bool   `json:"permanent_status"`
// 	Status          string `json:"approve_status"`
// 	CadreID         int    `json:"cadre_id"`
// 	EmployeeGroup   string `json:"employee_group" `
// 	Remarks         string `json:"remarks"`
// 	CadreName       string `json:"cadre_name"`
// 	PayLevel        int    `json:"pay_level"`
// 	GradePay        int    `json:"grade_pay"`
// 	DesignationId   int    `json:"designation_id"`
// }

// // NewPostManagementFetchResponse creates a new PostManagementMasterResponse instance
// func NewPostManagementFetchResponse(post *domain.PostManagementMaster) PostManagementFetchResponse {
// 	return PostManagementFetchResponse{
// 		OfficeID:        post.OfficeID,
// 		OfficeName:      post.OfficeName,
// 		PostID:          post.PostID,
// 		PostName:        post.PostName,
// 		GroupId:         post.GroupId,
// 		Designation:     post.Designation,
// 		FilledStatus:    post.FilledStatus,
// 		PermanentStatus: post.PermanentStatus,
// 		Status:          post.Status,
// 		CadreID:         post.CadreID,
// 		EmployeeGroup:   post.EmployeeGroup,
// 		Remarks:         post.Remarks,
// 		CadreName:       post.CadreName,
// 		PayLevel:        post.PayLevel,
// 		GradePay:        post.GradePay,
// 		DesignationId:   post.DesignationId,
// 	}
// }

// // PostManagementResponse represents the response structure for PostManagementMaster
// type PostManagementResponse struct {
// 	PostManagementID int    `json:"postmanagement_id"`
// 	OfficeID         int    `json:"office_id"`
// 	PostID           int    `json:"post_id"`
// 	PostName         string `json:"post_name"`
// 	OfficeName       string `json:"office_name"`
// 	Status           string `json:"status"`
// }

// // NewPostManagementMasterResponse creates a new PostManagementMasterResponse instance
// func NewPostManagementResponses(posts []domain.PostManagementMaster) []PostManagementResponse {
// 	var responses []PostManagementResponse

// 	for _, post := range posts {
// 		response := PostManagementResponse{
// 			PostManagementID: post.PostManagementID,
// 			OfficeID:         post.OfficeID,
// 			PostID:           post.PostID,
// 			PostName:         post.PostName,
// 			OfficeName:       post.OfficeName,
// 			Status:           post.Status,
// 		}
// 		responses = append(responses, response)
// 	}

// 	return responses
// }

// // PostManagementFilledStatusResponse represents the response structure for PostManagementMaster
// type PostManagementFilledStatusResponse struct {
// 	OfficeID      int    `json:"office_id"`
// 	PostID        int    `json:"post_id"`
// 	PostName      string `json:"post_name"`
// 	CadreName     string `json:"cadre_name"`
// 	Designation   string `json:"designation"`
// 	FilledStatus  string `json:"filled_status"`
// 	Status        string `json:"approve_status"`
// 	GroupID       int    `json:"group_id"`
// 	EmployeeGroup string `json:"employee_group"`
// 	CadreID       int    `json:"cadre_id"`
// }

// // NewPostManagementFetchResponse creates a new PostManagementMasterResponse instance
// func NewPostManagementFilledStatusResponse(post *domain.PostManagementMaster) PostManagementFilledStatusResponse {
// 	return PostManagementFilledStatusResponse{
// 		OfficeID:      post.OfficeID,
// 		PostID:        post.PostID,
// 		PostName:      post.PostName,
// 		CadreName:     post.CadreName,
// 		Designation:   post.Designation,
// 		FilledStatus:  post.FilledStatus,
// 		Status:        post.Status,
// 		GroupID:       post.GroupId,
// 		EmployeeGroup: post.EmployeeGroup,
// 		CadreID:       post.CadreID,
// 	}
// }

// // PostManagementGroupByCadreResponse represents the response structure for PostManagementMaster
// type PostManagementGroupByCadreResponse struct {
// 	Count        int    `json:"count"`
// 	CadreName    string `json:"cadre_name"`
// 	FilledStatus *string `json:"filled_status"`
// }

// // NewPostManagementGroupByCadreResponse creates a new PostManagementGroupByCadreResponse instance
// func NewPostManagementGroupByCadreResponse(post *domain.PostManagementMaster) PostManagementGroupByCadreResponse {
// 	return PostManagementGroupByCadreResponse{
// 		Count:        post.Count,
// 		CadreName:    post.CadreName,
// 		FilledStatus: &post.FilledStatus,
// 	}
// }

type DesignationMasterResponse struct {
	Designation   string `json:"designation"`
	GroupName     string `json:"group_name"`
	CadreName     string `json:"cadre_name"`
	CadreId       int    `json:"cadre_id"`
	GroupId       int16  `json:"group_id"`
	DesignationId int    `json:"designation_id"`
}

func NewDesignationMasterResponse(designation *domain.DesignationMaster) DesignationMasterResponse {
	return DesignationMasterResponse{
		Designation:   designation.Designation,
		GroupName:     designation.GroupName,
		CadreName:     designation.CadreName,
		CadreId:       designation.CadreId,
		GroupId:       designation.GroupId,
		DesignationId: designation.DesignationID,
	}
}

type PostMngmtEstablishmentResponse struct {
	OfficeID                  int       `json:"office_id"`
	OfficeName                string    `json:"office_name"`
	EstablishmentRegisterID   int       `json:"establishment_register_id"`
	EstablishmentRegisterName string    `json:"establishment_register_name"`
	Status                    string    `json:"status" `
	CreatedBy                 string    `json:"created_by" validate:"required"`
	CreatedAt                 time.Time `json:"created_date" validate:"required"`
}

func NewPostMngmtEstablishmentResponse(post *domain.PostManagementMaster1) PostMngmtEstablishmentResponse {
	return PostMngmtEstablishmentResponse{
		OfficeID:                  post.OfficeID,
		OfficeName:                post.OfficeName,
		EstablishmentRegisterID:   post.EstablishmentRegisterID,
		EstablishmentRegisterName: post.EstablishmentRegisterName,
		Status:                    post.Status,
		CreatedBy:                 post.CreatedBy,
		CreatedAt:                 post.CreatedOn,
	}
}

// PostManagementUpdateResponse represents the response structure for PostManagementMaster
type PostManagementUpdateResponse struct {
	PostID   int64 `json:"post_id"`
	OfficeID int64 `json:"office_id"`
	//Remarks          string `json:"remarks"`
	Status string `json:"status"`
}

// NewPostManagementUpdateResponse creates a new PostManagementMasterResponse instance
func NewPostManagementUpdateResponse(post *domain.PostManagementMaster4) PostManagementUpdateResponse {
	return PostManagementUpdateResponse{
		PostID:   post.PostID,
		OfficeID: post.OfficeID,
		//	Remarks:          post.Remarks,
		Status: post.Status,
	}
}

// PostManagementUpdateResponse represents the response structure for PostManagementMaster
// type PostManagementApproveResponse1 struct {
// 	PostID   int `json:"post_id"`
// 	OfficeID int `json:"office_id"`
// 	//Remarks          string `json:"remarks"`
// 	Status string `json:"status"`
// }

// // NewPostManagementUpdateResponse creates a new PostManagementMasterResponse instance
// func NewPostManagementApproveResponse1(post *domain.PostManagementMaster) PostManagementApproveResponse1 {
// 	return PostManagementApproveResponse1{
// 		PostID:   post.PostID,
// 		OfficeID: post.OfficeID,
// 		//	Remarks:          post.Remarks,
// 		Status: post.Status,
// 	}
// }

type PostMulMapResShort struct {
	PostID                int    `json:"post_id"`
	PostMapID             string `json:"post_map_id"`
	PostMappingColumnName string `json:"post_mapping_column_name"`
	PostMapPostId         int32  `json:"post_map_post_id"`
}

// newUserResponse is a helper function to create a response body for handling user data
func NewempMulAuthResShort(ema domain.PosttoPostMap) PostMulMapResShort {
	return PostMulMapResShort{
		PostID:                ema.EmployeePostID,
		PostMapID:             ema.PostMapID,
		PostMappingColumnName: ema.PostMappingColumnName,
		PostMapPostId:         ema.PostMapPostId,
	}

}

type PostMapUpdateResponse struct {
	EmployeePostID int       `db:"employee_post_id" json:"employee_post_id"`
	UpdatedDate    time.Time `db:"updated_date" json:"updated_date"`
}

// NewPostMapUpdateResponse is a helper function to create a response body for handling update data
func NewPostMapUpdateResponse(ptp *domain.PosttoPostMap) PostMapUpdateResponse {
	return PostMapUpdateResponse{
		EmployeePostID: ptp.EmployeePostID,
		UpdatedDate:    ptp.UpdatedDate,
	}

}

type PostMappingMasterFetchResponse struct {
	PostMapID              string `json:"post_map_id"`
	PostMappingColumnName  string `json:"post_mapping_column_name"`
	PostMappingDescription string `json:"post_mapping_description"`
}

// NewPostMappingMasterFetchResponse is a helper function to create a response body for handling update data
func NewPostMappingMasterFetchResponse(ptp *domain.PostMapMaster) PostMappingMasterFetchResponse {
	return PostMappingMasterFetchResponse{
		PostMapID:              ptp.PostMapID,
		PostMappingColumnName:  ptp.PostMappingColumnName,
		PostMappingDescription: ptp.PostMappingDescription,
	}

}

// NewPostMapCreateResponse represents the response structure for creating a new post mapping detail.
type PostMapCreateResponse struct {
	EmployeePostID int `json:"employee_post_id"`
}

// NewPostMapCreateResponse creates a new instance of NewPostMapCreateResponse.
func NewPostMapCreateResponse(createResponse *domain.PosttoPostMap) PostMapCreateResponse {
	return PostMapCreateResponse{
		EmployeePostID: int(createResponse.EmployeePostID),
	}
}

type PostMapUpdateResponseArray struct {
	EmployeePostID int         `db:"employee_post_id" json:"employee_post_id"`
	UpdatedDate    time.Time   `db:"updated_date" json:"updated_date"`
	FieldUpdated   interface{} `json:"field_updated"`
	NewValue       interface{} `json:"new_value"`
}

// NewPostMapUpdateResponseArray is a helper function to create a response body array for handling update data
func NewPostMapUpdateResponseArray(ptps []domain.PosttoPostMap, fieldName string, fieldValue interface{}) []PostMapUpdateResponseArray {
	var responses []PostMapUpdateResponseArray
	for _, ptp := range ptps {
		response := PostMapUpdateResponseArray{
			EmployeePostID: ptp.EmployeePostID,
			UpdatedDate:    ptp.UpdatedDate,
			FieldUpdated:   fieldName,
			NewValue:       fieldValue,
		}
		responses = append(responses, response)
	}
	return responses
}

// type CadreMasterDetails struct {
// 	CadreID   int    `json:"cadre_id"`
// 	CadreName string `json:"cadre_name"`
// }

// func NewCadreMasterDetails(cadre *domain.CadreMaster) CadreMasterDetails {
// 	return CadreMasterDetails{
// 		CadreID:   cadre.CadreID,
// 		CadreName: cadre.CadreName,
// 	}
// }

type DesignationMasterDetails struct {
	DesignationID   int    `json:"designation_id"`
	DesignationName string `json:"designation_name"`
}

func NewDesignationMasterDetails(designation *domain.DesignationMaster) DesignationMasterDetails {
	return DesignationMasterDetails{
		DesignationID:   designation.DesignationID,
		DesignationName: designation.Designation,
	}
}

type DocumentResponse struct {
	PostID                 int       `json:"post_id"`
	OrderCasemark          string    `json:"order_casemark"`
	OrderDate              time.Time `json:"order_date"`
	DocumentName           string    `json:"document_name"`
	DocumentType           string    `json:"document_type"`
	DocumentSize           int       `json:"document_size"`
	DocumentApproverPostID string    `json:"document_approver_post_id"`
	DocumentUploadStatus   string    `json:"document_upload_status"`
	DocumentUploadedBy     string    `json:"document_uploaded_by"`
	DocumentUploadedDate   time.Time `json:"document_uploaded_date"`
	DocumentApprovedBy     string    `json:"document_approved_by"`
	DocumentApprovedDate   time.Time `json:"document_approved_date"`
	Remarks                string    `json:"remarks"`
	DocumentFilePath       string    `json:"document_file_path"`
}

// func newDocumentResponse(docDetails *domain.DocumentMaster) *DocumentResponse {
// 	return &DocumentResponse{
// 		PostID:                 docDetails.PostID,
// 		OrderCasemark:          docDetails.OrderCasemark,
// 		OrderDate:              docDetails.OrderDate,
// 		DocumentName:           docDetails.DocumentName,
// 		DocumentType:           docDetails.DocumentType,
// 		DocumentSize:           docDetails.DocumentSize,
// 		DocumentApproverPostID: docDetails.DocumentApproverPostID,
// 		DocumentUploadStatus:   docDetails.DocumentUploadStatus,
// 		DocumentUploadedBy:     docDetails.DocumentUploadedBy,
// 		DocumentUploadedDate:   docDetails.DocumentUploadedDate,
// 		DocumentApprovedBy:     docDetails.DocumentApprovedBy,
// 		DocumentApprovedDate:   docDetails.DocumentApprovedDate,
// 		Remarks:                docDetails.Remarks,
// 		DocumentFilePath:       docDetails.DocumentFilePath,
// 	}
// }

type PostManagementMakerResponse struct {
	PostManagementMakerID int    `json:"post_management_id"`
	OfficeID              int64  `json:"office_id"`
	PostID                int64  `json:"post_id"`
	PostName              string `json:"post_name"`
	OfficeName            string `json:"office_name"`
	Status                string `json:"status"`
}

func NewPostManagementMakerResponse(post *domain.PostManagementMaster3) PostManagementMakerResponse {
	return PostManagementMakerResponse{
		PostManagementMakerID: post.PostManagementMakerID,
		OfficeID:              post.OfficeID,
		PostID:                post.PostID,
		PostName:              post.PostName,
		OfficeName:            post.OfficeName,
		Status:                post.Status,
	}
}

type PostManagementMakerFetchResponse struct {
	OfficeID        int       `json:"office_id"`
	OfficeName      string    `json:"office_name"`
	PostID          int       `json:"post_id"`
	PostName        string    `json:"post_name"`
	GroupId         int       `json:"group_id"`
	Designation     string    `json:"designation"`
	FilledStatus    string    `json:"filled_status"`
	PermanentStatus bool      `json:"permanent_status"`
	ApproveStatus   string    `json:"approve_status"`
	Status          string    `json:"status"`
	Remarks         string    `json:"remarks"`
	NewOfficeID     int       `json:"new_office_id"`
	NewOfficeName   string    `json:"new_office_name"`
	ExchangePostID  int       `json:"exchange_post_id"`
	OrderDate       time.Time `json:"order_date"`
	CadreID         int       `json:"cadre_id"`
	CadreName       string    `json:"cadre_name"`
	EmployeeGroup   string    `json:"employee_group" `
}

func NewPostManagementMakerFetchResponse(post *domain.PostManagementMaker) PostManagementMakerFetchResponse {
	return PostManagementMakerFetchResponse{
		OfficeID:        post.OfficeID,
		PostName:        post.PostName,
		OfficeName:      post.OfficeName,
		GroupId:         post.GroupId,
		FilledStatus:    post.FilledStatus,
		PostID:          post.PostID,
		Designation:     post.Designation,
		PermanentStatus: post.PermanentStatus,
		ApproveStatus:   post.ApproveStatus,
		Status:          post.Status,
		Remarks:         post.Remarks,
		NewOfficeID:     post.NewOfficeID,
		NewOfficeName:   post.NewOfficeName,
		ExchangePostID:  post.ExchangePostID,
		OrderDate:       post.OrderDate,
		CadreID:         post.CadreID,
		CadreName:       post.CadreName,
		EmployeeGroup:   post.EmployeeGroup,
	}
}

type PostMapUpdateResponseArrayForMultipleFields struct {
	EmployeePostID int         `db:"employee_post_id" json:"employee_post_id"`
	UpdatedDate    time.Time   `db:"updated_date" json:"updated_date"`
	FieldUpdated   string      `json:"field_updated"`
	NewValue       interface{} `json:"new_value"`
}

// NewPostMapUpdateResponseArrayForMultipleFields is a helper function to create a response body array for handling update data
func NewPostMapUpdateResponseArrayForMultipleFields(ptps []domain.PosttoPostMap) []PostMapUpdateResponseArrayForMultipleFields {
	var responses []PostMapUpdateResponseArrayForMultipleFields

	for _, ptp := range ptps {
		response := PostMapUpdateResponseArrayForMultipleFields{
			EmployeePostID: ptp.EmployeePostID,
			UpdatedDate:    ptp.UpdatedDate,
			FieldUpdated:   ptp.FieldUpdated,
			NewValue:       ptp.NewValue,
		}
		responses = append(responses, response)
	}

	return responses
}

// Response structure
type DocumentResponse1 struct {
	Message  string          `json:"message"`
	Filename string          `json:"filename"`
	Document domain.Document `json:"document"`
}

// Function to create a new document response
func newDocumentResponse1(doc domain.Document) DocumentResponse1 {
	return DocumentResponse1{
		Message:  "Document uploaded successfully",
		Filename: doc.DocumentName,
		Document: doc,
	}
}

// Define the response struct
type PostManagementDeleteResponse struct {
	Message string `json:"message"`
}

// Function to create a new response
func NewPostManagementDeleteResponse(Success string) PostManagementDeleteResponse {

	return PostManagementDeleteResponse{
		Message: Success,
	}
}

type PostManagementApproveResponse struct {
	Message string `json:"message"`
}

// Function to create a new response
func NewPostManagementApproveResponse(Success string) PostManagementApproveResponse {

	return PostManagementApproveResponse{
		Message: Success,
	}
}

type AuthorityDetailsResponse struct {
	Message          string                             `json:"message"`
	AuthorityDetails map[string]domain.AuthorityDetails `json:"authority_details"`
}

// func newAuthorityDetailsResponse(details map[string]domain.AuthorityDetails) AuthorityDetailsResponse {
// 	return AuthorityDetailsResponse{
// 		Message:          "Authority details retrieved successfully",
// 		AuthorityDetails: details,
// 	}
// }

type AuthorityDetailsResponse1 struct {
	EmployeePostID   int    `json:"employee_post_id"`
	EmployeeOfficeID int    `json:"employee_office_id"`
	ApproveStatus    string `json:"approve_status"`
	FieldName        string `json:"field_name"`
	FieldValue       int32  `json:"field_value"`
	DesignationName  string `json:"designation_name"`
	OfficeName       string `json:"office_name"`
}

// AuthorityDetailsResponseWithMessage holds the response with a single message and a slice of details.
type AuthorityDetailsResponseWithMessage struct {
	Message string                      `json:"message"`
	Details []AuthorityDetailsResponse1 `json:"details"`
}

// newAuthorityDetailsResponse1 creates an AuthorityDetailsResponseWithMessage with a single message and a list of details.
// func newAuthorityDetailsResponse1(details []domain.PosttoPostMap1) AuthorityDetailsResponseWithMessage {
// 	var responseDetails []AuthorityDetailsResponse1

// 	// Transform each PosttoPostMap1 to AuthorityDetailsResponse1
// 	for _, detail := range details {
// 		responseDetails = append(responseDetails, AuthorityDetailsResponse1{
// 			EmployeePostID:   detail.EmployeePostID,
// 			EmployeeOfficeID: detail.EmployeeOfficeID,
// 			ApproveStatus:    detail.ApproveStatus,
// 			FieldName:        detail.FieldName,
// 			FieldValue:       int32(detail.FieldValue),
// 			DesignationName:  detail.DesignationName,
// 			OfficeName:       detail.OfficeName,
// 		})
// 	}

// 	return AuthorityDetailsResponseWithMessage{
// 		Message: "Authority details retrieved successfully",
// 		Details: responseDetails,
// 	}
// }

type SurplusFetchResponse struct {
	OfficeID        int    `json:"office_id"`
	OfficeName      string `json:"office_name"`
	PostID          int    `json:"post_id"`
	PostName        string `json:"post_name"`
	GroupId         int    `json:"group_id"`
	Designation     string `json:"designation"`
	FilledStatus    string `json:"filled_status"`
	PermanentStatus bool   `json:"permanent_status"`
	Status          string `json:"current_status"`
	CadreID         int    `json:"cadre_id"`
	EmployeeGroup   string `json:"employee_group" `
	Remarks         string `json:"remarks"`
	CadreName       string `json:"cadre_name"`
	PayLevel        int    `json:"pay_level"`
	DesignationId   int    `json:"designation_id"`
}

// NewPostManagementFetchResponse creates a new PostManagementMasterResponse instance
func NewSurplusFetchResponse(post *domain.PostManagementMaker) SurplusFetchResponse {
	return SurplusFetchResponse{
		OfficeID:        post.OfficeID,
		OfficeName:      post.OfficeName,
		PostID:          post.PostID,
		PostName:        post.PostName,
		GroupId:         post.GroupId,
		Designation:     post.Designation,
		FilledStatus:    post.FilledStatus,
		PermanentStatus: post.PermanentStatus,
		Status:          post.Status,
		CadreID:         post.CadreID,
		EmployeeGroup:   post.EmployeeGroup,
		Remarks:         post.Remarks,
		CadreName:       post.CadreName,
		PayLevel:        post.PayLevel,
		DesignationId:   post.DesignationId,
	}
}

// // PostManagementFetchResponse represents the response structure for PostManagementMaster
// type PostManagementFetchVacantResponse struct {
// 	OfficeID      int    `json:"office_id"`
// 	PostID        int    `json:"post_id"`
// 	PostName      string `json:"post_name"`
// 	GroupId       int    `json:"group_id"`
// 	EmployeeGroup string `json:"employee_group" `
// 	DesignationId int    `json:"designation_id"`
// 	Designation   string `json:"designation"`
// 	CadreID       int    `json:"cadre_id"`
// 	CadreName     string `json:"cadre_name"`
// 	FilledStatus  string `json:"filled_status"`
// 	OfficeName    string `json:"office_name"`
// }

// // NewPostManagementFetchResponse creates a new PostManagementMasterResponse instance
// func NewPostManagementFetchVacantResponse(post *domain.PostManagementMaster) PostManagementFetchVacantResponse {
// 	return PostManagementFetchVacantResponse{
// 		OfficeID:      post.OfficeID,
// 		PostID:        post.PostID,
// 		PostName:      post.PostName,
// 		GroupId:       post.GroupId,
// 		EmployeeGroup: post.EmployeeGroup,
// 		DesignationId: post.DesignationId,
// 		Designation:   post.Designation,
// 		CadreID:       post.CadreID,
// 		CadreName:     post.CadreName,
// 		FilledStatus:  post.FilledStatus,
// 		OfficeName:    post.OfficeName,
// 	}
// }
