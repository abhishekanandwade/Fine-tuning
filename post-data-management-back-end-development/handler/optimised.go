package handler

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"path"
	"pmdm/core/domain"
	"pmdm/core/port"
	"pmdm/handler/response"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/volatiletech/null/v9"
	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
	apiutility "gitlab.cept.gov.in/it-2.0-common/api-utility"
	validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
)

// EstblishnentRegisterByOfficeReq represents a request body for listing post management master details
type EstblishnentRegisterByOfficeReq1 struct {
	OfficeID int `uri:"office_id" validate:"required"`
}

// EstblishnentRegisterByOffice godoc
//
//	@Summary		Get Establishment Register by Office ID
//	@Description	Fetches the establishment register details for a specific office ID.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			office-id	path		int										true	"Office ID"
//	@Success		200		{object}	response.EstblishnentRegisterByOfficeAPIResponse	"Establishment register details retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/office-establishment-register/{office-id}/establishments [get]
func (pmh *PostManagementHandler) EstblishnentRegisterByOfficeHandler(ctx *gin.Context) {
	var req PostManagementListWithOfficeIDRequest
	if err := ctx.ShouldBindUri(&req); err != nil {
		log.Error(ctx, "Binding failed for PostManagementListWithOfficeIDRequest", err.Error())
		apierrors.HandleBindingError(ctx, err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		log.Error(ctx, "Validation failed for PostManagementListWithOfficeIDRequest", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}

	// Query for a single record by office ID
	post, err := pmh.svc.EstblishnentRegisterByOfficeQuery(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "EstblishnentRegisterByOfficeQuery repo call failed", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Create the response
	rsp := response.NewEstblishnentRegisterByOfficeResponse(post)

	// Create metadata for a single record response
	metadata := port.NewMetaDataResponse(0, 1, 1) // Skip=0, Limit=1, TotalRecordsCount=1

	// Prepare the API response
	apiRsp := response.EstblishnentRegisterByOfficeAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 []response.EstblishnentRegisterByOfficeResponse{rsp}, // Wrap single response in a slice
	}

	log.Debug(ctx, "EstblishnentRegisterByOfficeHandler Response: %s", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
}

// UpdatePostManagementMasterRequest represents a request body for updating PostManagementMaster details
type UpdatePostManagementMasterRequest struct {
	PostID     int           `uri:"post-id" validate:"required"`
	OfficeID   int           `json:"office_id" validate:"required"`
	FieldName  []string      `json:"field_name" validate:"required"`
	FieldValue []interface{} `json:"field_value" validate:"required"`
	// Remarks          string        `json:"remarks" `
}

type ApprovePostManagementMasterRequest struct {
	PostID     int           `uri:"post-id" validate:"required"`
	OfficeID   int           `json:"office_id" validate:"required"`
	FieldName  []string      `json:"field_name" validate:"required"`
	FieldValue []interface{} `json:"field_value" validate:"required"`
	// Remarks          string        `json:"remarks" `
}

type DeletePostManagementMasterRequest1 struct {
	PostID int `uri:"post-id" validate:"required"`
}

type EstablishmentRegisterRequest struct {
	OfficeID                  int       `json:"office_id" validate:"required"`
	OfficeName                string    `json:"office_name" validate:"required"`
	EstablishmentRegisterName string    `json:"establishment_register_name" validate:"required"`
	CreatedBy                 string    `json:"created_by" validate:"required"`
	CreatedAt                 time.Time `json:"created_date" validate:"required"`
	Status                    string    `json:"status" validate:"required"`
}

// CreateEstablishmentRegister godoc
//
//	@Summary		Create Establishment Register
//	@Description	Creates a new establishment register with the provided details.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			body	body		EstablishmentRegisterRequest	true	"Details for creating an establishment register"
//	@Success		201		{object}	response.CreateEstablishmentRegisterAPIResponse	"Establishment register created successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse				"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse				"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse				"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse				"Data not found error"
//	@Failure		500		{object}	apierrors.APIErrorResponse				"Internal server error"
//	@Router			/post-management/office-establishment-register [post]
func (pmh *PostManagementHandler) CreateEstablishmentRegisterHandler(ctx *gin.Context) {
	var req EstablishmentRegisterRequest

	// Bind JSON request body
	if err := ctx.ShouldBindJSON(&req); err != nil {
		log.Error(ctx, "Binding failed for EstablishmentRegisterRequest: %s", err)
		apierrors.HandleBindingError(ctx, err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for EstablishmentRegisterRequest: %s", err)
		return
	}

	log.Debug(ctx, Requestpassedvalidation)

	// Prepare the establishment register object
	establishmentRegister := domain.PostManagementMaster{
		OfficeID:                  null.Int32From(int32(req.OfficeID)),
		OfficeName:                null.StringFrom(req.OfficeName),
		CreatedBy:                 null.StringFrom(req.CreatedBy),
		CreatedOn:                 null.TimeFrom(req.CreatedAt),
		Status:                    null.StringFrom(req.Status),
		EstablishmentRegisterName: null.StringFrom(req.EstablishmentRegisterName),
	}

	// Call service to create the establishment register
	register, err := pmh.svc.CreateEstablishmentRegister(ctx, establishmentRegister)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "CreateEstablishmentRegister Repo call failed: %s", err.Error())
		return
	}

	// Create the response
	rsp := response.NewCreateEstablishmentRegisterResponse(register)
	apiRsp := response.CreateEstablishmentRegisterAPIResponse{
		StatusCodeAndMessage: port.CreateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementByOfficeIDAndStatusHandler resposne: %v", apiRsp)
	// Send the success response
	handleCreateSuccess(ctx, apiRsp)
	log.Debug(ctx, "Successfully created establishment register")
}

type PostManagementMakerCreateRequest struct {
	PostID             int       `json:"post_id" validate:"required"`
	PostName           string    `json:"post_name" validate:"required"`
	OfficeID           int       `json:"office_id" validate:"required"`
	OfficeName         string    `json:"office_name" validate:"required"`
	NewOfficeID        int       `json:"new_office_id" validate:"required"`
	NewOfficeName      string    `json:"new_office_name" validate:"required"`
	CadreID            int       `json:"cadre_id" validate:"required"`
	GroupID            int       `json:"group_id" validate:"required"`
	Designation        string    `json:"designation" validate:"required"`
	GradePay           int       `json:"grade_pay"`
	Status             string    `json:"status" validate:"required"`
	OrderCaseMark      string    `json:"order_casemark" validate:"required"`
	OrderDate          time.Time `json:"order_date" validate:"required"`
	UploadOrderDocName string    `json:"upload_order_doc_name" validate:"required"`
	Remarks            string    `json:"remarks"`
	ApprovePostID      string    `json:"approve_post_id"`
	PayLevel           int       `json:"pay_level"`
	CadreName          string    `json:"cadre_name"`
	DesignationId      int       `json:"designation_id"`
	EmployeeGroup      string    `json:"employee_group"`
	ExchangePostID     int       `json:"exchange_post_id"`
	CreatedBy          string    `json:"created_by"`
}

type PostManagementMakerCreateRequests struct {
	PostCreateReq []PostManagementMakerCreateRequest `json:"postcreatereq"`
}

// CreatePostManagementMaker godoc
//
//	@Summary		Create Post Management Maker Records
//	@Description	Creates new post management maker records.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			PostManagementMakerCreateRequests	body		PostManagementMakerCreateRequests	true	"Request body for creating post management maker records"
//	@Success		201	{object}	response.CreatePostManagementMakerAPIResponse	"Post Management Maker Records Created Successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409	{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500	{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/makers [post]
func (pmh *PostManagementHandler) CreatePostManagementMakerHandler(ctx *gin.Context) {
	log.Debug(ctx, "Received request to create post management maker records")

	var reqs PostManagementMakerCreateRequests

	// Bind JSON request body
	if err := ctx.ShouldBindJSON(&reqs); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementMakerCreateRequests: %s", err)
		return
	}
	log.Debug(ctx, SuccessJSONBound, "request", reqs)

	// Validate the request
	if err := validation.ValidateStruct(reqs); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementMakerCreateRequests: %s", err)
		return
	}

	var createPosts []domain.PostManagementMaker
	for _, postreq := range reqs.PostCreateReq {
		createPost := domain.PostManagementMaker{
			OfficeID:           postreq.OfficeID,
			PostID:             postreq.PostID,
			PostName:           postreq.PostName,
			OfficeName:         postreq.OfficeName,
			NewOfficeID:        postreq.NewOfficeID,
			NewOfficeName:      postreq.NewOfficeName,
			GradePay:           postreq.GradePay,
			GroupId:            postreq.GroupID,
			OrderCaseMark:      postreq.OrderCaseMark,
			OrderDate:          postreq.OrderDate,
			Status:             postreq.Status,
			UploadOrderDocName: postreq.UploadOrderDocName,
			Designation:        postreq.Designation,
			CadreID:            postreq.CadreID,
			Remarks:            postreq.Remarks,
			ApproveStatus:      "Pending",
			ApprovePostID:      postreq.ApprovePostID,
			PayLevel:           postreq.PayLevel,
			CadreName:          postreq.CadreName,
			DesignationId:      postreq.DesignationId,
			EmployeeGroup:      postreq.EmployeeGroup,
			ExchangePostID:     postreq.ExchangePostID,
			CreatedBy:          postreq.CreatedBy,
		}
		createPosts = append(createPosts, createPost)
		log.Debug(ctx, "Prepared post management maker record for insertion", "post", createPost)
	}

	// Call service to create posts
	createResponse, err := pmh.svc.CreatePostManagementMakerQuery(ctx, createPosts)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "CreatePostManagementMakerQuery Repo call failed: %s", err.Error())
		return
	}
	log.Debug(ctx, "Successfully created post management maker records")

	// Prepare the response
	rsp := response.NewCreatePostManagementMakerResponse(createResponse)
	apiRsp := response.CreatePostManagementMakerAPIResponse{
		StatusCodeAndMessage: port.CreateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "CreatePostManagementMakerHandler resposne: %v", apiRsp)
	// Send the success response
	handleCreateSuccess(ctx, apiRsp)
	log.Debug(ctx, "Successfully handled create post management maker request")
}

type PostManagementListRequest21 struct {
	OfficeID int   `uri:"office-id" validate:"required"`
	RAPostID int64 `uri:"post-id" validate:"required"`
}

type PostManagementMakerApproveRequest struct {
	PostIDs    []int  `json:"post-ids" validate:"required"`
	ApprovedBy string `json:"approved-by" validate:"required"`
}

// ApprovePostManagementMaker godoc
//
//	@Summary		Approve Post Management Maker Records
//	@Description	Approves the selected post management maker records.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			PostManagementMakerApproveRequest	body		PostManagementMakerApproveRequest	true	"Request body for approving post management maker records"
//	@Success		200	{object}	response.ApprovePostManagementMakerAPIResponse	"Post Management Maker Records Approved Successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse										"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse										"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse										"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse										"Data not found error"
//	@Failure		409	{object}	apierrors.APIErrorResponse										"Data conflict error"
//	@Failure		500	{object}	apierrors.APIErrorResponse										"Internal server error"
//	@Router			/post-management/makers/approve-bulk [post]
func (pmh *PostManagementHandler) ApprovePostManagementMakerHandler(ctx *gin.Context) {
	log.Debug(ctx, "Received request to approve post management maker records")

	var req PostManagementMakerApproveRequest

	// Bind query parameters
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, BindingFailed+"%s", err)
		return
	}

	// Bind JSON request body
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, BindingFailed, err)
		return
	}

	log.Debug(ctx, SuccessJSONBound, "request", req)

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, ValidationFailed+"%s", err)
		return
	}
	log.Debug(ctx, Requestpassedvalidation)

	// Call the service to approve the post management maker records
	_, err := pmh.svc.ApprovePostManagementMakerQuery(ctx, req.PostIDs, req.ApprovedBy)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "ApprovePostManagementMakerQuery Repo call failed: %s", err.Error())
		return
	}

	// Create the response with metadata
	rsp := response.NewApprovePostManagementMakerResponse(req.PostIDs, req.ApprovedBy)
	apiRsp := response.ApprovePostManagementMakerAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "ApprovePostManagementMakerHandler resposne: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
	log.Debug(ctx, "Successfully handled approve post management maker request")
}

type PostManagementMasterListRequest1 struct {
	PostID string `uri:"approved-by" validate:"required"` //approved-by
}

type PostManagementMakerListRequest struct {
	PostID string `form:"post-id" validate:"required"`
}

type PostManagementMakerListRequest2 struct {
	PostID string `uri:"post-id" validate:"required"`
}

type PendingCreatePostApprovalRequest struct {
	OfficeID int `form:"office-id" validate:"required"`
}

// PostManagementWithPendingStatusOfMaker godoc
//
//	@Summary		Get Post Management with Pending Status of Maker
//	@Description	Fetches post management details that are in pending status for a specific Maker's Post ID
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id	query		int										true	"Post ID of the Maker"
//	@Success		200		{object}	response.PostManagementWithPendingStatusOfMakerAPIResponse	"Post management details with pending status retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse						"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse						"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse						"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse						"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse						"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse						"Internal server error"
//	@Router			/post-management/makers/status-pending [get]
func (pmh *PostManagementHandler) PostManagementWithPendingStatusOfMakerHandler(ctx *gin.Context) {
	var req PostManagementMakerListRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementMakerListRequest: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementMakerListRequest2: %s", err)
		return
	}

	postList, err := pmh.svc.PostManagementWithPendingStatusOfMakerQuery(ctx, req.PostID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementWithPendingStatusOfMakerQuery Repo call failed: %s", err.Error())
		return
	}

	// Prepare the response
	rsp := response.NewPostManagementWithPendingStatusOfMakerResponse(postList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.PostManagementWithPendingStatusOfMakerAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementWithPendingStatusOfMakerHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// FetchPendingCreatePostApprovalsByOfficeID godoc
//
//	@Summary		Fetch Pending Create Post Approvals by Office ID
//	@Description	Fetches post management maker records with pending approval status and New Post status for a specific office ID
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			office-id	query		int										true	"Office ID"
//	@Success		200		{object}	response.FetchPostsByOfficeIDAndMakerAPIResponse	"Pending create post approvals retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/makers/approve-create-post [get]
func (pmh *PostManagementHandler) FetchPendingCreatePostApprovalsByOfficeIDHandler(ctx *gin.Context) {
	var req PendingCreatePostApprovalRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PendingCreatePostApprovalRequest: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PendingCreatePostApprovalRequest: %s", err)
		return
	}

	postList, err := pmh.svc.PostManagementWithPendingCreatePostByOfficeID(ctx, req.OfficeID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementWithPendingCreatePostByOfficeID Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewFetchPostsByOfficeIDAndMakerResponse(postList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.FetchPostsByOfficeIDAndMakerAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "FetchPendingCreatePostApprovalsByOfficeIDHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// ApprovePostManagementMakerForAbolishPost godoc
//
//	@Summary		Approve Post Management Maker for Abolishment
//	@Description	Approves and abolishes the post management maker records based on PostID, ApprovedBy, and ApproveStatus.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			PostManagementMakerApproveRequest	body		PostManagementMakerApproveRequest	true	"Approve Post Management Maker for Abolishment details"
//	@Success		200	{object}	response.ApprovePostManagementMakerForAbolishPostAPIResponse	"Post Management Maker for Abolishment approved successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse										"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse										"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse										"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse										"Data not found error"
//	@Failure		500	{object}	apierrors.APIErrorResponse										"Internal server error"
//	@Router			/post-management/makers/abolish-bulk [post]
func (pmh *PostManagementHandler) ApprovePostManagementMakerForAbolishPostHandler(ctx *gin.Context) {
	log.Debug(ctx, "Received request to approve and abolish post management maker records")

	var req PostManagementMakerApproveRequest

	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, BindingFailed+"%s", err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, ValidationFailed+"%s", err)
		return
	}

	// Prepare the office details map for constructing the response
	officeDetails := make(map[int]struct {
		NewOfficeID   int
		NewOfficeName string
		OfficeID      int
		Status        string
	})

	// Call the service to approve and abolish the posts
	_, err := pmh.svc.ApprovePostManagementMakerForAbolishPost(ctx, req.PostIDs, req.ApprovedBy)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "ApprovePostManagementMakerForAbolishPost Repo call failed: %s", err.Error())
		return
	}

	// Create the response
	rsp := response.NewApprovePostManagementMakerForAbolishPostResponse(req.PostIDs, req.ApprovedBy, officeDetails)
	apiRsp := response.ApprovePostManagementMakerForAbolishPostAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "ApprovePostManagementMakerForAbolishPostHandler resposne: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
	log.Debug(ctx, "Successfully handled approve post management maker for abolish post request")
}

type PostManagementMasterChangeRequest struct {
	PostID int `uri:"post-id" validate:"required"`
}

// PostManagementChangFilledStatusByPostID godoc
//
//	@Summary		Change Filled Status by Post ID
//	@Description	Changes the filled status of a post by Post ID.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id		path		int		true	"ID of the post to change the filled status"
//	@Success		200	{object}	response.PostManagementChangFilledStatusAPIResponse	"Filled status changed successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse										"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse										"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse										"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse										"Post not found"
//	@Failure		500	{object}	apierrors.APIErrorResponse										"Internal server error"
//	@Router			/post-management/posts/{post-id}/status-change [put]
func (pmh *PostManagementHandler) PostManagementChangFilledStatusByPostIDHandler(ctx *gin.Context) {
	var req PostManagementMasterChangeRequest

	// Bind URI parameters
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementMasterChangeRequest: %s", err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementMasterChangeRequest: %s", err)
		return
	}

	// Call the service to change the filled status by PostID
	message, err := pmh.svc.PostManagementChangFilledStatusByPostID(ctx, req.PostID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementChangFilledStatusByPostID Repo call failed: %s", err.Error())
		return
	}

	// Create the response data
	rsp := response.NewPostManagementChangFilledStatusResponse(message)
	apiRsp := response.PostManagementChangFilledStatusAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementChangFilledStatusByPostIDHandler resposne: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
}

// func (emh *PostManagementHandler) UploadFile(ctx *gin.Context) {
// 	fmt.Println("in uploadFile Handler")
// 	folderPath := ctx.PostForm("folderPath")
// 	file, header, err := ctx.Request.FormFile("file")
// 	if err != nil {
// 		ctx.String(http.StatusBadRequest, "Bad request")
// 		return
// 	}
// 	defer func(file multipart.File) {
// 		err := file.Close()
// 		if err != nil {
// 			//log.Print("Error in closing file")
// 		}
// 	}(file)

// 	filename := strings.ToLower(header.Filename)
// 	filename = strings.ReplaceAll(filename, " ", "_")
// 	filename = strings.ReplaceAll(filename, "-", "_")

// 	objectName := path.Join(folderPath, filename)
// 	err = emh.svc.UploadFile(file, objectName, header.Header.Get("Content-Type"), header.Size)
// 	if err != nil {
// 		ctx.String(http.StatusInternalServerError, "Failed to upload file")
// 		return
// 	}

//		ctx.JSON(http.StatusOK, gin.H{"key": objectName})
//	}
type UploadFileRequest struct {
	OfficeID   int    `json:"office_id" validate:"required,gte=10000000,lte=99999999"`
	FolderPath string `json:"folderPath" validate:"required"`
}

// UploadFile godoc
//
//	@Summary		Upload File
//	@Description	Allows uploading a file and storing its metadata in the database.
//	@Tags			Post Management
//	@Accept			multipart/form-data
//	@Produce		json
//	@Param			data		formData	string	true	"JSON payload containing metadata"
//	@Param			file		formData	file	true	"File to be uploaded"
//	@Success		200			{object}	DocumentResponse1	"File uploaded successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse	"Bad Request: Invalid or missing parameters"
//	@Failure		500			{object}	apierrors.APIErrorResponse	"Internal Server Error: Unable to upload file"
//	@Router			/post-management/files/upload [post]
func (emh *PostManagementHandler) UploadFile(ctx *gin.Context) {
	log.Debug(ctx, "Starting UploadFile handler...")

	// Retrieve and unmarshal JSON data from form data
	jsonData := ctx.PostForm("data")
	var req UploadFileRequest

	if err := json.Unmarshal([]byte(jsonData), &req); err != nil {
		log.Error(ctx, "JSON Unmarshal Error: ", err.Error())
		apierrors.HandleMarshalError(ctx, err)
		return
	}

	log.Debug(ctx, "Parsed JSON data successfully: ", jsonData)

	file, header, err := ctx.Request.FormFile("file")
	if err != nil {
		log.Error(ctx, "Failed to get file from request: ", err.Error())
		ctx.String(http.StatusBadRequest, "Bad request: file missing")
		return
	}
	defer func(file multipart.File) {
		if err := file.Close(); err != nil {
			log.Error(ctx, "Failed to close file: ", err.Error())
			apierrors.HandleFileTypeError(ctx)
		}
	}(file)

	log.Debug(ctx, "Received file: ", header.Filename)

	fileType := header.Header.Get("Content-Type")
	log.Debug(ctx, "File type: ", fileType)

	ext := path.Ext(header.Filename)
	uniqueID := generateUniqueID() // Implement this to generate a unique identifier
	filename := fmt.Sprintf("%d_%s%s", req.OfficeID, uniqueID, ext)

	objectName := path.Join(req.FolderPath, filename)
	log.Debug(ctx, "Generated object name: ", objectName)

	// Call repository to upload the file
	log.Debug(ctx, "Calling repository to upload file to MinIO...")
	err = emh.svc.UploadFile(ctx, file, objectName, fileType, header.Size)
	if err != nil {
		log.Error(ctx, "Failed to upload file to MinIO: ", err.Error())
		ctx.String(http.StatusInternalServerError, "Failed to upload file")
		return
	}
	log.Info(ctx, "File uploaded successfully to MinIO: ", objectName)

	// Insert file details into the database
	log.Debug(ctx, "Inserting file details into document_master_pmdm table...")
	document := domain.Document{
		OfficeID:             req.OfficeID,
		DocumentName:         filename,
		DocumentType:         fileType,
		DocumentSize:         header.Size,
		DocumentFilePath:     objectName,
		DocumentUploadStatus: "uploaded", // Example status
		DocumentUploadedBy:   "user_id",  // Retrieve the actual user ID from context or session
		DocumentUploadedDate: time.Now(),
	}
	err = emh.svc.InsertDocument(ctx, document)
	if err != nil {
		log.Error(ctx, "Failed to insert document record: ", err.Error())
		ctx.String(http.StatusInternalServerError, "Failed to insert document record")
		return
	}
	log.Info(ctx, "Document record inserted successfully.")

	// Construct and return the response
	log.Debug(ctx, "Constructing response...")
	rsp := newDocumentResponse1(document)
	handleSuccess(ctx, rsp)
}

// func (emh *PostManagementHandler) UploadFile(ctx *gin.Context) {
// 	// Retrieve and unmarshal JSON data from form data
// 	jsonData := ctx.PostForm("data")
// 	var req UploadFileRequest

// 	// Unmarshal JSON data into req
// 	if err := json.Unmarshal([]byte(jsonData), &req); err != nil {
// 		log.Error(ctx, "Unmarshal Error: ", err.Error())
// 		apierrors.HandleMarshalError(ctx, err)
// 		return
// 	}
// 	file, header, err := ctx.Request.FormFile("file")
// 	if err != nil {
// 		ctx.String(http.StatusBadRequest, "Bad request")
// 		return
// 	}
// 	defer func(file multipart.File) {
// 		err := file.Close()
// 		if err != nil {
// 			apierrors.HandleFileTypeError(ctx)
// 		}
// 	}(file)

// 	fileType := header.Header.Get("Content-Type")
// 	// if !allowedMimeTypes[fileType] {
// 	// 	apierrors.HandleErrorWithStatusCodeAndMessage(ctx, apierrors.AppErrorValidationError, "File Type is not allowed", nil)
// 	// 	return
// 	// }

// 	// Log step: Generate custom filename
// 	//log.Error(ctx, "Generating custom filename...")
// 	ext := path.Ext(header.Filename)
// 	uniqueID := generateUniqueID() // Implement this function to generate a unique number or UUID

// 	filename := fmt.Sprintf("%s_%s%s", req.OfficeID, uniqueID, ext)

// 	objectName := path.Join(req.FolderPath, filename)
// 	fmt.Println("before upload file repo")
// 	err = emh.svc.UploadFile(file, objectName, fileType, header.Size)
// 	if err != nil {
// 		ctx.String(http.StatusInternalServerError, "Failed to upload file")
// 		return
// 	}
// 	log.Error(ctx, "Uploading file on server is completed...")
// 	// Log step: Insert details into the document_master_pmdm table
// 	log.Error(ctx, "Inserting details into the document_master_pmdm table...")
// 	document := domain.Document{
// 		OfficeID:             req.OfficeID,
// 		DocumentName:         filename,
// 		DocumentType:         fileType,
// 		DocumentSize:         header.Size,
// 		DocumentFilePath:     objectName,
// 		DocumentUploadStatus: "uploaded", // Example status
// 		DocumentUploadedBy:   "user_id",  // Retrieve the actual user ID from context or session
// 		DocumentUploadedDate: time.Now(),
// 	}
// 	err = emh.svc.InsertDocument(ctx, document)
// 	if err != nil {
// 		log.Error(ctx, "Failed to insert document record: %v", err)
// 		ctx.String(http.StatusInternalServerError, "Failed to insert document record")
// 		return
// 	}

// 	// Log step: Construct response
// 	log.Debug(ctx, "Constructing response...")
// 	rsp := newDocumentResponse1(document)
// 	handleSuccess(ctx, rsp)
// }

// DownloadFiles godoc
// @Summary        Download Files as a ZIP for a specific Office ID
// @Description    Fetch document metadata for a given Office ID, download the files, and return them as a ZIP archive.
// @Tags           Post Management
// @Accept         json
// @Produce        application/zip
// @Param          office_id   query    integer true "Office ID (unique identifier for the office)"
// @Success        200         {file}   application/zip "ZIP file containing the documents"
// @Failure        400         {object} apierrors.APIErrorResponse "Bad Request: Invalid or missing Office ID"
// @Failure        404         {object} apierrors.APIErrorResponse "Not Found: No files found for the specified Office ID"
// @Failure        500         {object} apierrors.APIErrorResponse "Internal Server Error: Failed to fetch documents or generate ZIP"
// @Router         /post-management/files [get]
func (emh *PostManagementHandler) DownloadFiles(c *gin.Context) {
	// Validate office_id
	officeIDStr := c.Query("office_id")
	if officeIDStr == "" {
		log.Error(c, "Missing office_id parameter")
		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "office_id is required", errors.New("missing required parameter: office_id"))
		return
	}

	// Convert office_id to integer
	officeID, err := strconv.Atoi(officeIDStr)
	if err != nil {
		log.Error(c, "Invalid office_id format", "value", officeIDStr)
		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "office_id should be an integer", errors.New("office_id should be an integer"))
		return
	}

	ctx := c.Request.Context()
	log.Info(ctx, "Fetching documents for office_id", "office_id", officeID)

	// Fetch document metadata
	documents, err := emh.svc.GetDocumentsByOfficeID(ctx, officeID)
	if err != nil {
		log.Error(ctx, "Failed to fetch document metadata", "error", err)
		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "Failed to fetch document metadata", err)
		return
	}

	if len(documents) == 0 {
		log.Warn(ctx, "No files found for the given office_id", "office_id", officeID)
		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "No files found for the given office_id", errors.New("no files found"))
		return
	}

	// Create ZIP buffer
	buf := new(bytes.Buffer)
	zipWriter := zip.NewWriter(buf)

	// Track if at least one file is successfully added
	filesAdded := 0
	missingFiles := []string{}

	// Process each document
	for _, document := range documents {
		objectPath := strings.TrimSpace(document.DocumentFilePath)
		log.Info(ctx, "Processing file", "file_path", objectPath)

		if objectPath == "" {
			log.Warn(ctx, "Skipping empty document path", "document_name", document.DocumentName)
			continue
		}

		// Download file from MinIO
		object, err := emh.svc.DownloadFile(objectPath)
		if err != nil {
			log.Warn(ctx, "File not found in MinIO", "file", objectPath)
			missingFiles = append(missingFiles, objectPath)
			continue // Skip missing files but continue processing others
		}

		// Create ZIP entry
		zipFileWriter, err := zipWriter.Create(path.Base(objectPath))
		if err != nil {
			log.Error(ctx, "Failed to create zip entry", "file", objectPath, "error", err)
			continue
		}

		// Copy file content directly into ZIP
		if _, err := io.Copy(zipFileWriter, object); err != nil {
			log.Error(ctx, "Failed to write file to zip", "file", objectPath, "error", err)
			continue
		}

		filesAdded++
	}

	// If no files were successfully added, return an error instead of an empty ZIP
	if filesAdded == 0 {
		log.Error(ctx, "No valid files found for download")
		c.String(http.StatusNotFound, "No valid files found to download")
		return
	}

	// Ensure ZIP is closed before sending response
	if err := zipWriter.Close(); err != nil {
		log.Error(ctx, "Failed to close zip writer", "error", err)
		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "Failed to close zip writer", err)
		return
	}

	// Set response headers
	c.Writer.Header().Set("Content-Type", "application/zip")
	c.Writer.Header().Set("Content-Disposition", "attachment; filename=documentspmdm.zip")
	c.Writer.Header().Set("Content-Length", strconv.Itoa(buf.Len()))

	// Write ZIP data to response
	_, err = buf.WriteTo(c.Writer)
	if err != nil {
		log.Error(ctx, "Failed to write zip file to response", "error", err)
	}

	// Log missing files (for debugging)
	if len(missingFiles) > 0 {
		log.Warn(ctx, "Some files were missing and not included in the ZIP", "missing_files", missingFiles)
	}
}

// func (emh *PostManagementHandler) DownloadFiles(c *gin.Context) {
// 	// Validate office_id
// 	officeIDStr := c.Query("office_id")
// 	if officeIDStr == "" {
// 		log.Error(nil, "Missing office_id parameter")
// 		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "office_id is required", errors.New("missing required parameter: office_id"))
// 		return
// 	}

// 	// Convert office_id to integer
// 	officeID, err := strconv.Atoi(officeIDStr)
// 	if err != nil {
// 		log.Error(nil, "Invalid office_id format", "value", officeIDStr)
// 		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "office_id should be an integer", errors.New("office_id should be an integer"))
// 		return
// 	}

// 	ctx := c.Request.Context()
// 	log.Info(ctx, "Fetching documents for office_id", "office_id", officeID)

// 	// Fetch document metadata
// 	documents, err := emh.svc.GetDocumentsByOfficeID(ctx, officeID)
// 	if err != nil {
// 		log.Error(ctx, "Failed to fetch document metadata", "error", err)
// 		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "Failed to fetch document metadata", err)
// 		return
// 	}

// 	if len(documents) == 0 {
// 		log.Warn(ctx, "No files found for the given office_id", "office_id", officeID)
// 		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "No files found for the given office_id", errors.New("No files found"))
// 		return
// 	}

// 	// Create ZIP buffer
// 	buf := new(bytes.Buffer)
// 	zipWriter := zip.NewWriter(buf)

// 	// Process each document
// 	for _, document := range documents {
// 		objectPath := strings.TrimSpace(document.DocumentFilePath)
// 		log.Info(ctx, "Processing file", "file_path", objectPath)

// 		if objectPath == "" {
// 			log.Warn(ctx, "Skipping empty document path", "document_name", document.DocumentName)
// 			continue
// 		}

// 		// Download file from MinIO
// 		object, err := emh.svc.DownloadFile(objectPath)
// 		if err != nil {
// 			log.Warn(ctx, "File not found in MinIO", "file", objectPath)
// 			continue // Skip missing files
// 		}

// 		// Create ZIP entry
// 		zipFileWriter, err := zipWriter.Create(path.Base(objectPath))
// 		if err != nil {
// 			log.Error(ctx, "Failed to create zip entry", "file", objectPath, "error", err)
// 			continue
// 		}

// 		// Copy file content directly into ZIP
// 		if _, err := io.Copy(zipFileWriter, object); err != nil {
// 			log.Error(ctx, "Failed to write file to zip", "file", objectPath, "error", err)
// 			continue
// 		}
// 	}

// 	// Ensure ZIP is closed before sending response
// 	if err := zipWriter.Close(); err != nil {
// 		log.Error(ctx, "Failed to close zip writer", "error", err)
// 		apierrors.HandleErrorWithStatusCodeAndMessage(c, apierrors.AppErrorValidationError, "Failed to close zip writer", err)
// 		return
// 	}

// 	// If ZIP is empty, return 404
// 	if buf.Len() == 0 {
// 		c.String(http.StatusNotFound, "No files found to download")
// 		return
// 	}

// 	// ✅ **Set Response Headers BEFORE Writing Content**
// 	c.Writer.Header().Set("Content-Type", "application/zip")
// 	c.Writer.Header().Set("Content-Disposition", "attachment; filename=documentspmdm.zip")
// 	c.Writer.Header().Set("Content-Length", strconv.Itoa(buf.Len()))

// 	// ✅ **Ensure Response is Written**
// 	_, err = buf.WriteTo(c.Writer)
// 	if err != nil {
// 		log.Error(ctx, "Failed to write zip file to response", "error", err)
// 	}
// }

func RejectQueryParamsMiddleware(ctx *gin.Context) {
	if len(ctx.Request.URL.Query()) > 0 {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"status_code": 400,
			"message":     "Query parameters are not allowed for this endpoint",
			"success":     false,
		})
		ctx.Abort()
		return
	}
	ctx.Next()
}

type SurplusPostRequest struct {
	ApprovePostID string `uri:"post-id" validate:"required"`
}

// FetchSurplusPostRecordByApproverPostID godoc
//
//	@Summary		Get Surplus Post Records by Approver Post ID
//	@Description	Fetches surplus post records based on the provided Approver Post ID
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id	path		int										true	"Post ID"
//	@Success		200		{object}	response.FetchSurplusPostRecordByApproverPostIDAPIResponse	"Surplus post records retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse						"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse						"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse						"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse						"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse						"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse						"Internal server error"
//	@Router			/post-management/posts/{post-id}/surplus [get]
func (pmh *PostManagementHandler) FetchSurplusPostRecordByApproverPostIDHandler(ctx *gin.Context) {
	var req SurplusPostRequest
	// Reject the request if any query parameters are present
	if len(ctx.Request.URL.Query()) > 0 {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"status_code": 400,
			"message":     "Query parameters are not allowed for this endpoint",
			"success":     false,
		})
		return
	}
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for SurplusPostRequest: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for SurplusPostRequest: %s", err)
		return
	}

	postList, err := pmh.svc.FetchSurplusPostRecordByApproverPostID(ctx, req.ApprovePostID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "FetchSurplusPostRecordByApproverPostID Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewFetchSurplusPostRecordByApproverPostIDResponse(postList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.FetchSurplusPostRecordByApproverPostIDAPIResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "FetchSurplusPostRecordByApproverPostIDHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type RestoreApproveRequest struct {
	PostIDs   []int  `json:"post-ids" validate:"required"`
	UpdatedBy string `json:"updated-by" validate:"required"`
}

// RestoredSurplusPost godoc
//
//	@Summary		Restore Surplus Post
//	@Description	Restores surplus post management maker records based on the provided Post IDs.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			RestoreApproveRequest	body		RestoreApproveRequest	true	"Request body for restoring surplus posts including Post IDs and UpdatedBy"
//	@Success		200	{object}	response.RestoredSurplusPostAPIResponse	"Surplus post records restored successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409	{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500	{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/posts/surplus-restore-bulk [post]
func (pmh *PostManagementHandler) RestoredSurplusPostHandler(ctx *gin.Context) {
	log.Debug(ctx, "Received request to restore post management maker records")
	var req RestoreApproveRequest

	// Bind query parameters
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for RestoreApproveRequest: %s", err)
		return
	}

	// Bind JSON request body
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for RestoreApproveRequest: %s", err)
		return
	}

	log.Debug(ctx, SuccessJSONBound, "request", req)

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for RestoreApproveRequest: %s", err)
		return
	}
	log.Debug(ctx, Requestpassedvalidation)

	// Call the service to restore surplus posts
	approveResponse, err := pmh.svc.RestoredSurplusPost(ctx, req.PostIDs, req.UpdatedBy)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "RestoredSurplusPost Repo call failed: %s", err.Error())
		return
	}

	log.Debug(ctx, "Successfully restored post management maker records")

	// Prepare the response
	rsp := response.NewRestoredSurplusPostResponse(approveResponse)
	apiRsp := response.RestoredSurplusPostAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "RestoredSurplusPostHandler resposne: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
}

// RejectPostManagementMaker godoc
//
//	@Summary		Reject Post Management Maker
//	@Description	Rejects the post management maker records based on PostIDs and ApprovedBy.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			PostManagementMakerApproveRequest	body		PostManagementMakerApproveRequest	true	"Reject post management maker records"
//	@Success		200	{object}	response.RejectPostManagementMakerAPIResponse	"Post management maker records rejected successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse									"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse									"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse									"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse									"Data not found error"
//	@Failure		500	{object}	apierrors.APIErrorResponse									"Internal server error"
//	@Router			/post-management/makers/reject-bulk [post]
func (pmh *PostManagementHandler) RejectPostManagementMakerHandler(ctx *gin.Context) {
	log.Debug(ctx, "Received request to reject post management maker records")

	var req PostManagementMakerApproveRequest

	// Bind JSON request body
	if err := ctx.ShouldBindJSON(&req); err != nil {
		log.Error(ctx, BindingFailed+"%s", err)
		apierrors.HandleBindingError(ctx, err)
		return
	}
	log.Debug(ctx, SuccessJSONBound, "request", req)

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		log.Error(ctx, ValidationFailed+"%s", err)
		apierrors.HandleValidationError(ctx, err)
		return
	}
	log.Debug(ctx, Requestpassedvalidation)

	// Call the service layer to reject the post management maker records
	_, err := pmh.svc.RejectPostManagementMaker(ctx, req.PostIDs, req.ApprovedBy)
	if err != nil {
		log.Error(ctx, "RejectPostManagementMaker Repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	log.Debug(ctx, "Successfully rejected post management maker records")

	// Create the response
	rsp := response.NewRejectPostManagementMakerResponse(req.PostIDs, req.ApprovedBy)

	apiRsp := response.RejectPostManagementMakerAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "RejectPostManagementMakerHandler resposne: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
	log.Debug(ctx, "Successfully handled reject post management maker request")
}

// ApprovePostManagementMakerForExchangePost godoc
//
//	@Summary		Approve Post Management Maker for Exchange Post
//	@Description	Approves post management maker records for exchange posts based on PostIDs and ApprovedBy.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			PostManagementMakerApproveRequest	body		PostManagementMakerApproveRequest	true	"Approve post management maker records for exchange"
//	@Success		200	{object}	response.ApprovePostManagementMakerForExchangePostAPIResponse	"Post management maker records approved successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse										"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse										"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse										"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse										"Data not found error"
//	@Failure		500	{object}	apierrors.APIErrorResponse										"Internal server error"
//	@Router			/post-management/makers/exchange-post-bulk [post]
func (pmh *PostManagementHandler) ApprovePostManagementMakerForExchangePostHandler(ctx *gin.Context) {
	log.Debug(ctx, "Received request to approve post management maker records")

	var req PostManagementMakerApproveRequest

	// Bind JSON request body
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, BindingFailed+"%s", err)
		return
	}
	log.Debug(ctx, SuccessJSONBound, "request", req)

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, ValidationFailed+"%s", err)
		return
	}
	log.Debug(ctx, Requestpassedvalidation)

	// Approve the post management maker records for exchange
	_, err := pmh.svc.ApprovePostManagementMakerForExchangePost(ctx, req.PostIDs, req.ApprovedBy)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "ApprovePostManagementMakerForExchangePost Repo call failed: %s", err.Error())
		return
	}
	log.Debug(ctx, "Successfully approved post management maker records")

	// Create the response
	responseData := response.NewApprovePostManagementMakerForExchangePostResponse(req.PostIDs, req.ApprovedBy)

	apiRsp := response.ApprovePostManagementMakerForExchangePostAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 responseData,
	}

	log.Debug(ctx, "ApprovePostManagementMakerForExchangePostHandler resposne: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)

}

type PostManagementMakerApproveOrRejectRequest struct {
	PostIDs       []int  `json:"post_ids" validate:"required"`
	ApprovedBy    string `json:"approved_by" validate:"required"`
	ApproveStatus string `json:"approve_status" validate:"required"`
	Remarks       string `json:"remarks" validate:"required"`
}

// ApprovePostManagementMasterWithMaker godoc
//
//	@Summary		Approve or Reject Post Management Maker Records
//	@Description	Approves or rejects post management maker records based on the provided details.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			PostManagementMakerApproveOrRejectRequest	body		PostManagementMakerApproveOrRejectRequest	true	"Request body for approving or rejecting maker records"
//	@Success		200	{object}	response.ApprovePostManagementMasterWithMakerAPIResponse	"Successfully processed the request"
//	@Failure		400	{object}	apierrors.APIErrorResponse					"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse					"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse					"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse					"Data not found error"
//	@Failure		500	{object}	apierrors.APIErrorResponse					"Internal server error"
//	@Router			/post-management/posts/approve [post]
func (pmh *PostManagementHandler) ApprovePostManagementMasterWithMaker(ctx *gin.Context) {
	log.Debug(ctx, "Received request to approve/reject post management maker records")
	var req PostManagementMakerApproveOrRejectRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, ErrorBinding, err)
		return
	}
	log.Debug(ctx, SuccessJSONBound, "request", req)

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}
	log.Debug(ctx, Requestpassedvalidation)

	approveResponse, err := pmh.svc.ApprovePostManagementMasterWithMaker(ctx, req.PostIDs, req.ApprovedBy, req.ApproveStatus, req.Remarks)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "Error approving/rejecting post management maker records: ", err)
		return
	}
	log.Debug(ctx, "Successfully processed post management maker records")
	rsp := response.NewApprovePostManagementMasterWithMakerResponse(approveResponse)
	apiRsp := response.ApprovePostManagementMasterWithMakerAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}
	log.Debug(ctx, "ApprovePostManagementMasterWithMaker resposne: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
}

// GetPostManagementMasterWithMaker godoc
//
//	@Summary		Get Post Management Master with Maker Details
//	@Description	Fetches the post management master details along with maker information for a specific Post ID
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id	path		int										true	"Post ID"
//	@Success		200		{object}	response.GetPostManagementMasterWithMakerAPIResponse	"Post management master details retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse						"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse						"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse						"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse						"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse						"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse						"Internal server error"
//	@Router			/post-management/posts/{post-id} [get]
func (pmh *PostManagementHandler) GetPostManagementMasterWithMakerHandler(ctx *gin.Context) {
	var req PostManagementMakerListRequest2
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed forPostManagementMakerListRequest: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementMakerListRequest: %s", err)
		return
	}

	postList, err := pmh.svc.GetPostManagementMasterWithMaker(ctx, req.PostID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "GetPostManagementMasterWithMaker Repo call failed: %s", err.Error())
		return
	}
	rsp := response.NewGetPostManagementMasterWithMakerResponse(postList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.GetPostManagementMasterWithMakerAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetPostManagementMasterWithMakerHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type PostManagementWithEstablishmentRequest1 struct {
	EstablishmentRegisterID int `uri:"register-id" validate:"required"`
}

type PostNameMasterRequest struct {
	CadreID int `uri:"cadre-id" validate:"required"`
	port.MetaDataRequest
}

// GetPostNameMaster godoc
//
//	@Summary		Get Post Names by Cadre ID
//	@Description	Fetches post names based on the provided Cadre ID
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			cadre-id	path		string										true	"Cadre ID"
//	@Param       metaDataRequest	query		port.MetaDataRequest	false  		"Metadata request containing skip, limit, orderBy & sortBy"
//	@Success		200			{object}	response.GetPostNameMasterAPIResponse		"Post names retrieved successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401			{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403			{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404			{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409			{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500			{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/cadres/{cadre-id} [get]
func (pmh *PostManagementHandler) GetPostNameMasterHandler(ctx *gin.Context) {
	var req PostNameMasterRequest
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostNameMasterRequest: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostNameMasterRequest: %s", err)
		return
	}
	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}

	fmt.Println(req.CadreID)
	postList, err := pmh.svc.GetPostNameMaster(ctx, req.CadreID, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "GetPostNameMaster Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewGetPostNameMasterResponse(postList)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.GetPostNameMasterAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetPostNameMasterHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type EstablishmentRequest1 struct {
	OfficeID   int    `uri:"office-id" validate:"required" json:"-"`
	OfficeType string `uri:"office-type" validate:"required" json:"-"`
}

type DesignationMasterListRequest struct {
	GroupCode string `form:"group-code" validate:"required"`
	CadreCode string `form:"cadre-code" validate:"required"`
	port.MetaDataRequest
}

// PostMappingDetailUpdateRequest represents a request body for updating post mapping details
type PostMappingDetailUpdateRequest1 struct {
	EmpPostID  int           `uri:"post-id" validate:"required"`
	FieldName  []string      `json:"field_name" validate:"required"`
	FieldValue []interface{} `json:"field_value" validate:"required"`
}

// GetPostMappingMaster godoc
//
//	@Summary		Get Post Mapping Master Records
//	@Description	Fetches all post mapping master records.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Success		200		{object}	response.GetPostMappingMasterAPIResponse	"Post mapping master records retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/ptop-mappings [get]
func (pph *PosttoPostMappingrHandler) GetPostMappingMasterHandler(ctx *gin.Context) {
	// Fetch post mapping master records
	fetchResponse, err := pph.svc.GetPostMappingMasterQuery(ctx)
	if err != nil {
		log.Error(ctx, "GetPostMappingMasterQuery repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Create the response data
	rsp := response.NewGetPostMappingMasterResponse(fetchResponse)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.GetPostMappingMasterAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetPostMappingMasterHandler Response: %s", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
}

type PostMappingDetailCreateRequest1 struct {
	EmpPostID   int `json:"employee_post_id" validate:"required"`
	EmpOfficeID int `json:"employee_office_id" validate:"required"`
}

type UpdateArrayOfEmpPostIDForParticularFieldRequest1 struct {
	OfficeID   int         `json:"office-id" validate:"required"`
	EmpPostID  []int       `json:"employee_post_id" validate:"required"`
	FieldName  string      `json:"field_name" validate:"required"`
	FieldValue interface{} `json:"field_value" validate:"required"`
}

// UpdateArrayOfEmpPostIDForParticularField godoc
//
//	@Summary		Update Array of EmpPostID for a Particular Field
//	@Description	Updates a specific field for a given array of Employee Post IDs.
//	@Tags			Post Mapping
//	@Accept			json
//	@Produce		json
//	@Param			UpdateArrayOfEmpPostIDForParticularFieldRequest1	body		UpdateArrayOfEmpPostIDForParticularFieldRequest1	true	"Request to update a specific field for array of EmpPostID"
//	@Success		200	{object}	response.UpdateArrayOfEmpPostIDAPIResponse	"Post Mapping Field Updated Successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409	{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500	{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/ptop-mappings/posts-bulk [post]
func (pmh *PosttoPostMappingrHandler) UpdateArrayOfEmpPostIDForParticularField(ctx *gin.Context) {
	log.Debug(ctx, "Received request to update array of EmpPostID for a particular field")

	var req UpdateArrayOfEmpPostIDForParticularFieldRequest1

	if err := ctx.ShouldBindUri(&req); err != nil {
		log.Error(ctx, "Binding failed for UpdateArrayOfEmpPostIDForParticularFieldRequest1: %s", err.Error())
		apierrors.HandleBindingError(ctx, err)
		return
	}

	if err := ctx.ShouldBindJSON(&req); err != nil {
		log.Error(ctx, "Binding failed for UpdateArrayOfEmpPostIDForParticularFieldRequest1: %s", err.Error())
		apierrors.HandleBindingError(ctx, err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}
	log.Debug(ctx, "Validation successful")

	updateResponse, err := pmh.svc.UpdateArrayOfEmpPostIDForParticularFieldQuery(
		ctx, req.EmpPostID, req.FieldName, req.FieldValue, req.OfficeID,
	)
	if err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Error updating post mapping: ", err)
		return
	}

	rsp := response.NewPostMapUpdateResponseArray(updateResponse, req.FieldName, req.FieldValue)
	apiRsp := response.UpdateArrayOfEmpPostIDAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}
	log.Debug(ctx, "UpdateArrayOfEmpPostIDForParticularField response:%v", rsp)
	handleSuccess(ctx, apiRsp)
}

type ListAuthRequest struct {
	PostID int32 `form:"post-id" validate:"required"`
	port.MetaDataRequest
}

type ListAuthDetailsRequest struct {
	PostID int32 `uri:"post-id" validate:"required"`
	port.MetaDataRequest
}

// GetAuthorityDetailsByPostID godoc
//
//	@Summary		Get Authority Details by Post ID
//	@Description	Fetches authority details for a specific Post ID.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id	path		int										true	"Post ID"
//	@Param       metaDataRequest	query		port.MetaDataRequest	false  		"Metadata request containing skip, limit, orderBy & sortBy"
//
// @Success		200		{object}	response.GetAuthorityDetailsByPostIDAPIResponse	"Authority details retrieved successfully"
//
//	@Failure		400		{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/ptop-mappings/{post-id}/authority-details [get]
func (pph *PosttoPostMappingrHandler) GetAuthorityDetailsByPostIDHandler(ctx *gin.Context) {
	var req ListAuthDetailsRequest

	// Bind URI parameters
	if err := ctx.ShouldBindUri(&req); err != nil {
		log.Error(ctx, "Binding failed for ListAuthRequest: %s", err.Error())
		apierrors.HandleBindingError(ctx, err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		log.Error(ctx, "Validation failed for ListAuthRequest: %s", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}
	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}
	// Fetch authority details by PostID
	authrows, err := pph.svc.GetAuthorityDetailsByPostID(ctx, req.PostID, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "GetAuthorityDetailsByPostID repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Create the response data
	rsp := response.NewGetAuthorityDetailsByPostIDResponse(authrows)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.GetAuthorityDetailsByPostIDAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetAuthorityDetailsByPostIDHandler Response: %s", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
}

// type getAuthforMultiple1 struct {
// 	PostIDs []int `form:"post_id" validate:"required"`
// }

func (pph *PosttoPostMappingrHandler) GetAuthorityDetailsForMultiplePostID(ctx *gin.Context) {
	var req getAuthforMultiple
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Debug(ctx, Error+" %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	authRowsMap, err := pph.svc.GetAuthorityDetailsForMultiplePostID(ctx, req.PostIDs)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	handleSuccess(ctx, authRowsMap)
}

type UpdateArrayOfEmpPostIDForManyFieldRequest1 struct {
	OfficeID   []int         `json:"office-id" validate:"required"`
	EmpPostID  []int         `json:"employee-post-id" validate:"required"`
	FieldName  []string      `json:"field-name" validate:"required"`
	FieldValue []interface{} `json:"field-value" validate:"required"`
}

// CreatePostMappingDetailMaker godoc
//
//	@Summary		Create Post Mapping Detail Maker
//	@Description	Creates a new post mapping detail for the maker.
//	@Tags			Post Mapping
//	@Accept			json
//	@Produce		json
//	@Param			MultipleEmpPostIDForMultipleFieldRequest	body		MultipleEmpPostIDForMultipleFieldRequest	true	"Request to create post mapping detail"
//	@Success		201	{object}	response.CreatePostMappingDetailMakerAPIResponse	"Post Mapping Detail Created Successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409	{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500	{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/ptop-mappings-makers [post]
func (pmh *PosttoPostMappingrHandler) CreatePostMappingDetailMaker(ctx *gin.Context) {
	log.Debug(ctx, "Received request to create post mapping detail maker")

	var req MultipleEmpPostIDForMultipleFieldRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		validationError(ctx, err)
		log.Error(ctx, ErrorBinding, err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	log.Debug(ctx, "Validation successful")

	// Call the service to create post mapping detail maker
	updateResponse, err := pmh.svc.CreatePostMappingDetailMaker(
		ctx, req.EmpPostID, req.FieldName, req.FieldValue, req.OfficeID, req.PostID, req.CreatedBy)
	if err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Error creating post mapping detail maker: ", err)
		return
	}

	// Prepare the response using the helper function
	rsp := response.NewPostMapCreateResponseArray(updateResponse)
	apiRsp := response.CreatePostMappingDetailMakerAPIResponse{
		StatusCodeAndMessage: port.CreateSuccess,
		Data:                 rsp,
	}

	// Handle the success response
	handleCreateSuccess(ctx, apiRsp)
	log.Debug(ctx, "Successfully created post mapping detail maker")
}

// GetPostMappingMasterMaker godoc
//
//	@Summary		Get Post Mapping Master Maker Details
//	@Description	Fetches post mapping master maker details for a specific Post ID.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id	query		int										true	"Post ID"
//	@Success		200		{object}	response.GetPostMappingMasterMakerAPIResponse	"Post mapping master maker details retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/ptop-mappings-makers [get]
func (pph *PosttoPostMappingrHandler) GetPostMappingMasterMaker(ctx *gin.Context) {
	var req ListAuthRequest

	// Bind query parameters
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Debug(ctx, Error+" %s", err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	// Fetch authority details
	authrows, err := pph.svc.GetPostMappingMasterMaker(ctx, req.PostID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Create the response data
	rsp := response.NewGetPostMappingMasterMakerResponse(authrows)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.GetPostMappingMasterMakerAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	// Send the success response
	handleSuccess(ctx, apiRsp)
}

type ApprovePostMappingDetailRequest1 struct {
	EmployeePostID int    `json:"post-id" validate:"required"`
	ApprovedBy     string `json:"approved_by" validate:"required"`
	FieldName      string `json:"field_name" validate:"required" `
	FieldValue     int32  `json:"field_value" `
	Status         string `json:"approve_status"  `
	OfficeID       int    `json:"authority_office_id" `
	Remarks        string `json:"remarks"  `
}

// ApprovePostMappingDetailMaker godoc
//
//	@Summary		Approve Post Mapping Detail Maker
//	@Description	Approves post mapping details for a given list of requests.
//	@Tags			Post Mapping
//	@Accept			json
//	@Produce		json
//	@Param			ApprovePostMappingDetailRequest	body		[]ApprovePostMappingDetailRequest				true	"List of Post Mapping Details to be approved"
//	@Success		200	{object}	response.ApprovePostMappingDetailMakerAPIResponse	"Post Mapping Details Approved"
//	@Failure		400	{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409	{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500	{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/ptop-mappings-makers/approve-bulk [post]
func (pmh *PosttoPostMappingrHandler) ApprovePostMappingDetailMaker(ctx *gin.Context) {
	log.Debug(ctx, "Received request to approve post mapping detail maker")

	var reqs []ApprovePostMappingDetailRequest
	// if err := ctx.ShouldBindUri(&reqs); err != nil {
	// 	apierrors.HandleBindingError(ctx, err)
	// 	validationError(ctx, err)
	// 	log.Error(ctx, "Error binding URI request: ", err)
	// 	return
	// }

	if err := ctx.ShouldBindJSON(&reqs); err != nil {
		apierrors.HandleBindingError(ctx, err)
		validationError(ctx, err)
		log.Error(ctx, ErrorBinding, err)
		return
	}

	// Validate each request
	for _, req := range reqs {
		if err := validation.ValidateStruct(req); err != nil {
			apierrors.HandleValidationError(ctx, err)
			return
		}
	}

	// Convert requests to domain model
	var details []domain.ApprovePostMappingDetail
	for _, req := range reqs {
		details = append(details, domain.ApprovePostMappingDetail{
			EmployeePostID: req.EmployeePostID,
			OfficeID:       req.OfficeID,
			FieldName:      req.FieldName,
			FieldValue:     req.FieldValue,
			Status:         req.Status,
			ApprovedBy:     req.ApprovedBy,
			Remarks:        req.Remarks,
		})
	}

	// Call the service layer to process the approval
	approveResponse, err := pmh.svc.ApprovePostMappingDetailMaker(ctx, details)
	if err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Error approving post mapping detail maker: ", err)
		return
	}

	// Prepare the response array with the appropriate structure
	// convertedDetail := domain.ApprovePostMappingDetail{
	// 	EmployeePostID: approveResponse.EmployeePostID,
	// 	OfficeID:       approveResponse.OfficeID,
	// 	FieldName:      approveResponse.FieldUpdated,
	// 	FieldValue:     approveResponse.NewValue.(int32), // Ensure proper casting
	// 	Status:         approveResponse.ApproveStatus,
	// 	ApprovedBy:     approveResponse.UpdatedBy,
	// 	Remarks:        "Approved and updated",
	// }

	// Wrap the response into a slice
	rsp := response.NewApprovePostMappingDetailMakerResponse(approveResponse)

	// Prepare the API response
	apiRsp := response.ApprovePostMappingDetailMakerAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	// Send the success response
	handleCreateSuccess(ctx, apiRsp)
	log.Debug(ctx, "Successfully approved post mapping detail maker")
}

type GetMasterAuthRequest struct {
	PostID int `uri:"post-id" validate:"required"`
}

// GetMasterAuthoritiesDeatils godoc
//
//	@Summary		Get Master Authorities Details by Post ID
//	@Description	Fetches master authorities details for a specific Post ID.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id	path		int										true	"Post ID"
//	@Success		200		{object}	response.GetMasterAuthoritiesDeatilsAPIResponse	"Master authorities details retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/ptop-mappings/{post-id}/master-authority-details [get]
func (pph *PosttoPostMappingrHandler) GetMasterAuthoritiesDeatilsHandler(ctx *gin.Context) {
	var req GetMasterAuthRequest

	// Bind URI parameters
	if err := ctx.ShouldBindUri(&req); err != nil {
		log.Error(ctx, "Binding failed for GetMasterAuthRequest: %s", err.Error())
		apierrors.HandleBindingError(ctx, err)
		log.Debug(ctx, Error+" %s", err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		log.Error(ctx, "Validation failed for GetMasterAuthRequest: %s", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}

	// Fetch authority details based on PostID
	authrows, err := pph.svc.GetMasterAuthoritiesDeatils(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "GetMasterAuthoritiesDeatils repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Create the response data
	rsp := response.NewGetMasterAuthoritiesDeatilsResponse(authrows)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.GetMasterAuthoritiesDeatilsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	// Send the success response
	handleSuccess(ctx, apiRsp)
}

type GetPostMappingMakerDeatilsAuth struct {
	PostID string `uri:"approve-post-id" validate:"required"`
	port.MetaDataRequest
}

// GetPostMappingMakerDetails retrieves the details of post mapping maker based on the Post ID and metadata.
//
// @Summary Retrieve post mapping maker details
// @Description Fetches the details of post mapping maker for a given Post ID and additional metadata.
// @Tags Post Management
// @Accept json
// @Produce json
// @Param approve-post-id path string true "Post ID for which details are to be fetched"
// @Param limit query int false "Limit the number of records returned"
// @Param offset query int false "Offset for pagination"
// @Param sort query string false "Sort order for the results"
// @Success 200 {object} response.GetPostMappingMakerDetailsAPIResponse "Successful response with metadata and data"
// @Failure 400 {object} apierrors.APIErrorResponse "Invalid request parameters"
// @Failure 500 {object} apierrors.APIErrorResponse "Internal server error"
// @Router /post-management/ptop-mappings-makers/{approve-post-id} [get]
func (pph *PosttoPostMappingrHandler) GetPostMappingMakerDetails(ctx *gin.Context) {
	var req GetPostMappingMakerDeatilsAuth
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Debug(ctx, Error+" %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}

	authrows, err := pph.svc.GetPostMappingMakerDetails(ctx, req.PostID, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}
	// Create the response data
	rsp := response.NewGetPostMappingMakerDetailsResponse(authrows)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.GetPostMappingMakerDetailsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	// Send the success response
	handleSuccess(ctx, apiRsp)
}

// // FetchPostsByOfficeID godoc
// //
// //	@Summary		Get Posts by Office ID
// //	@Description	Fetches posts for a specific Office ID. Can be filtered by post ID or vacancy status.
// //	@Tags			Post Management
// //	@Accept			json
// //	@Produce		json
// //	@Param			office-id	path		int										true	"Office ID"
// //	@Param			post-id		query		string									false	"Post ID (optional)"
// //	@Param			vacant		query		string									false	"Vacant status (true/false). Default is false."
// //	@Success		200			{object}	response.FetchPostsByOfficeIDAPIResponse	"Posts retrieved successfully"
// //	@Failure		400			{object}	apierrors.APIErrorResponse						"Validation error"
// //	@Failure		401			{object}	apierrors.APIErrorResponse						"Unauthorized error"
// //	@Failure		403			{object}	apierrors.APIErrorResponse						"Forbidden error"
// //	@Failure		404			{object}	apierrors.APIErrorResponse						"Data not found error"
// //	@Failure		409			{object}	apierrors.APIErrorResponse						"Data conflict error"
// //	@Failure		500			{object}	apierrors.APIErrorResponse						"Internal server error"
// //	@Router			/post-management/office-post-details/{office-id}/posts-summary [get]
// func (pmh *PostManagementHandler) FetchPostsByOfficeIDHandler(ctx *gin.Context) {
// 	var req PostManagementListRequest123
// 	if err := ctx.ShouldBindUri(&req); err != nil {
// 		apierrors.HandleBindingError(ctx, err)
// 		log.Error(ctx, "Binding failed for PostManagementListRequest123: %s", err)
// 		return
// 	}
// 	if err := validation.ValidateStruct(req); err != nil {
// 		apierrors.HandleValidationError(ctx, err)
// 		log.Error(ctx, "Validation failed for PostManagementListRequest123: %s", err)
// 		return
// 	}

// 	postIDStr := ctx.Query("post-id")
// 	isVacant := ctx.DefaultQuery("vacant", "false")

// 	var masterList []domain.PostManagementMaster
// 	var makerList []domain.PostManagementMaker
// 	var err error

// 	if postIDStr != "" {
// 		postID, err := strconv.ParseInt(postIDStr, 10, 64)
// 		if err != nil {
// 			apierrors.HandleErrorWithStatusCodeAndMessage(ctx, apierrors.AppErrorValidationError, "invalid post-id", err)
// 			return
// 		}
// 		makerList, err = pmh.svc.PostManagementByOfficeIDMDWMaker(ctx, req.OfficeID, postID)
// 		if err != nil {
// 			apierrors.HandleDBError(ctx, err)
// 			log.Error(ctx, "PostManagementByOfficeIDMDWMaker Repo call failed: %s", err.Error())
// 			return
// 		}
// 	} else if isVacant == "true" {
// 		masterList, err = pmh.svc.FetchVacantActivePostByOfficeID(ctx, req.OfficeID)
// 		if err != nil {
// 			apierrors.HandleDBError(ctx, err)
// 			log.Error(ctx, "FetchVacantActivePostByOfficeID Repo call failed: %s", err.Error())
// 			return
// 		}
// 	} else {
// 		masterList, err = pmh.svc.FetchAllActivePostByOfficeID(ctx, req.OfficeID)
// 		if err != nil {
// 			apierrors.HandleDBError(ctx, err)
// 			log.Error(ctx, "FetchAllActivePostByOfficeID Repo call failed: %s", err.Error())
// 			return
// 		}
// 	}

// 	rsp := response.NewFetchPostsByOfficeIDResponse(masterList, makerList)
// 	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
// 	apiRsp := response.FetchPostsByOfficeIDAPIResponse{
// 		StatusCodeAndMessage: port.ListSuccess,
// 		MetaDataResponse:     metadata,
// 		Data:                 rsp,
// 	}

// 	log.Debug(ctx, "FetchPostsByOfficeIDHandler resposne: %v", apiRsp)
// 	handleSuccess(ctx, apiRsp)
// }

// FetchPostsByOfficeIDAndMaker godoc
//
//	@Summary		Get Posts by Office ID and Maker
//	@Description	Fetches posts for a specific Office ID and Maker. Can be filtered by Post ID.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			office-id	query		int										true	"Office ID"
//	@Param			post-id		query		string									false	"Post ID (optional)"
//	@Success		200		{object}	response.FetchPostsByOfficeIDAndMakerAPIResponse	"Posts retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/makers [get]
func (pmh *PostManagementHandler) FetchPostsByOfficeIDAndMakerHandler(ctx *gin.Context) {
	var req PostManagementListWithOfficeIDForMakerRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementListWithOfficeIDForMakerRequest: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementListWithOfficeIDForMakerRequest: %s", err)
		return
	}

	var postList []domain.PostManagementMaker
	var err error

	if req.PostID != "" {
		postID, err := strconv.ParseInt(req.PostID, 10, 64)
		if err != nil {
			apierrors.HandleBindingError(ctx, errors.New("invalid post-id"))
			return
		}
		postList, err = pmh.svc.PostManagementByOfficeIDMDWMaker(ctx, req.OfficeID, postID)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "PostManagementByOfficeIDMDWMaker Repo call failed: %s", err.Error())
			return
		}
	} else {
		postList, err = pmh.svc.PostManagementWithApprovedStatusOfMakerByOfficeID(ctx, req.OfficeID)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "PostManagementWithApprovedStatusOfMakerByOfficeID Repo call failed: %s", err.Error())
			return
		}
	}

	rsp := response.NewFetchPostsByOfficeIDAndMakerResponse(postList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.FetchPostsByOfficeIDAndMakerAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "FetchPostsByOfficeIDAndMakerHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type EstablishmentRegisterQueryRequest struct {
	RegisterID string `form:"register-id"` // Optional query parameter
	OfficeID   string `form:"office-id"`   // Required if register-id is not provided
	OfficeType string `form:"office-type"` // Required if office-id is provided
}

// FetchEstablishmentRegister godoc
//
//	@Summary		Get Establishment Register
//	@Description	Fetches establishment register details by either register ID or a combination of office ID and office type.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			register-id	query		string									false	"Register ID (optional)"
//	@Param			office-id	query		string									false	"Office ID (required if register-id is not provided)"
//	@Param			office-type	query		string									false	"Office Type (required if office-id is provided)"
//	@Success		200			{object}	response.FetchEstablishmentRegisterAPIResponse	"Establishment register details retrieved successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse								"Validation error or missing parameters"
//	@Failure		401			{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403			{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404			{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409			{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500			{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/office-establishment-register [get]
func (pmh *PostManagementHandler) FetchEstablishmentRegisterHandler(ctx *gin.Context) {
	var req EstablishmentRegisterQueryRequest

	// Bind the query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for EstablishmentRegisterQueryRequest: %s", err)
		return
	}

	var postList []domain.PostManagementMaster
	var err error

	// Fetch data based on query parameters
	if req.RegisterID != "" {
		registerID, parseErr := strconv.Atoi(req.RegisterID)
		if parseErr != nil {
			apierrors.HandleBindingError(ctx, parseErr)
			log.Error(ctx, "Invalid register-id: %s", parseErr)
			return
		}
		postList, err = pmh.svc.PostManagementByEstablishmentRegisterID(ctx, registerID)
		if err != nil {
			log.Error(ctx, "PostManagementByEstablishmentRegisterID Repo call failed: %s", err.Error())
			apierrors.HandleDBError(ctx, err)
			return
		}
	} else if req.OfficeID != "" && req.OfficeType != "" {
		officeID, parseErr := strconv.Atoi(req.OfficeID)
		if parseErr != nil {
			apierrors.HandleErrorWithStatusCodeAndMessage(ctx, apierrors.AppErrorValidationError, "error in string conversion of office ID", parseErr)
			log.Error(ctx, "Invalid office-id: %s", parseErr)
			return
		}
		postList, err = pmh.svc.ViewEstablishmentRegisterByAuthority(ctx, officeID, req.OfficeType)
		if err != nil {
			log.Error(ctx, "ViewEstablishmentRegisterByAuthority Repo call failed: %s", err.Error())
			apierrors.HandleDBError(ctx, err)
			return
		}
	} else {
		log.Error(ctx, "Missing required parameters")
		apierrors.HandleError(ctx, errors.New("either register-id or a combination of office-id and office-type must be provided"))
		return
	}

	// Create the response data
	rsp := response.NewFetchEstablishmentRegisterResponse(postList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.FetchEstablishmentRegisterAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	// Send the success response
	log.Debug(ctx, "FetchEstablishmentRegisterHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// CreatePostManagementMaster godoc
//
//	@Summary        Create Post Management Master
//	@Description    Creates a post management master entry. Requires all mandatory fields like OfficeID, PostName, GroupId, etc.
//	@Tags           postManagement
//	@Accept         json
//	@Produce        json
//	@Param          PostManagementCreateRequests   body   PostManagementCreateRequests  true    "List of post management details to create"
//	@Success        200 {object} response.CreatePostManagementMasterAPIResponse "Post management master details created successfully"
//	@Failure        400 {object} apierrors.APIErrorResponse "Validation error"
//	@Failure        401 {object} apierrors.APIErrorResponse "Unauthorized access"
//	@Failure        403 {object} apierrors.APIErrorResponse "Forbidden access"
//	@Failure        404 {object} apierrors.APIErrorResponse "Resource not found"
//	@Failure        409 {object} apierrors.APIErrorResponse "Conflict in the data"
//	@Failure        500 {object} apierrors.APIErrorResponse "Internal server error"
//	@Router         /post-management/posts/create [post]
func (pmh *PostManagementHandler) CreatePostManagementMasterHandler(ctx *gin.Context) {
	var reqs PostManagementCreateRequests
	if err := ctx.ShouldBindJSON(&reqs); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementCreateRequests: %s", err)
		return
	}
	if err := validation.ValidateStruct(reqs); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementFetchRequest: %s", err)
		return
	}

	var createPosts []domain.PostManagementMasterNew1
	for _, postreq := range reqs.PostCreateReq {
		createPost := domain.PostManagementMasterNew1{
			OfficeID:                  null.Int32From(int32(postreq.OfficeID)),
			PostName:                  null.StringFrom(postreq.PostName),
			OfficeName:                null.StringFrom(postreq.OfficeName),
			GroupId:                   null.Int32From(int32(postreq.GroupId)),
			CadreID:                   null.Int32From(int32(postreq.CadreID)),
			CadreName:                 null.StringFrom(postreq.CadreName),
			AllowancesAttached:        null.BoolFrom(postreq.AllowancesAttached),
			AllowanceDescription:      null.StringFrom(postreq.AllowanceDescription),
			CreatedBy:                 null.StringFrom(postreq.CreatedBy),
			OrderCaseMark:             null.StringFrom(postreq.OrderCaseMark),
			OrderDate:                 null.TimeFrom(postreq.OrderDate),
			UploadOrderDocName:        null.StringFrom(postreq.UploadOrderDocName),
			EstablishmentRegisterID:   null.Int32From(int32(postreq.EstablishmentRegisterID)),
			Designation:               null.StringFrom(postreq.Designation),
			PayLevel:                  null.Int32From(int32(postreq.PayLevel)),
			GradePay:                  null.Int32From(int32(postreq.GradePay)),
			PermanentStatus:           null.BoolFrom(postreq.PermanentStatus),
			EstablishmentRegisterName: null.StringFrom(postreq.EstablishmentRegisterName),
			EmployeeGroup:             null.StringFrom(postreq.EmployeeGroup),
			SanctionedStrength:        null.Int32From(int32(postreq.SanctionedStrength)),
			ApprovePostID:             null.StringFrom(postreq.ApprovePostID),
			OfficeType:                null.StringFrom(postreq.OfficeType),
			GroupName:                 null.StringFrom(postreq.GroupName),
			DesignationId:             null.Int32From(int32(postreq.DesignationId)),
		}
		createPosts = append(createPosts, createPost)
	}
	createResponse, err := pmh.svc.CreatePostManagementMasterQuery(ctx, createPosts)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "CreatePostManagementMasterQuery Repo call failed: %s", err.Error())
		return
	}
	// Prepare the response
	rsp := response.NewCreatePostManagementMasterResponse(createResponse)
	apiRsp := response.CreatePostManagementMasterAPIResponse{
		StatusCodeAndMessage: port.CreateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "CreatePostManagementMakerHandler resposne: %v", apiRsp)
	// Send the success response
	handleCreateSuccess(ctx, apiRsp)
}

type PostManagementQueryRequest struct {
	CadreName  string `form:"cadre-name"`                    // Query parameter, included in binding
	OfficeID   int    `form:"office-id" validate:"required"` // Required query parameter, excluded from JSON
	PostID     int64  `form:"post-id"`                       // Optional query parameter, excluded from JSON
	ApprovedBy string `form:"approved-by"`                   // Optional query parameter, included in binding
	port.MetaDataRequest
}

// PostManagementByOfficeAndPost godoc
//
//	@Summary		Get Post Management by Office and Post
//	@Description	Retrieve post management details using various filters like Office ID, Post ID, Cadre Name, and Approved By.
//
// You can also paginate and sort the results using metadata parameters (skip, limit, order_by, and sort_type).
//
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			office-id	query		int											true	"Office ID (required)"
//	@Param			post-id		query		int64										false	"Post ID (optional)"
//	@Param			cadre-name	query		string										false	"Cadre Name (optional)"
//	@Param			approved-by	query		string										false	"Approved By (optional)"
//	@Param			skip		query		uint64										false	"Number of records to skip (default: 0)"
//	@Param			limit		query		uint64										false	"Number of records to retrieve (default: 10)"
//	@Param			order_by	query		string										false	"Field to sort by (e.g., id, name)"
//	@Param			sort_type	query		string										false	"Sort direction (asc or desc)"
//	@Param			total_records_required	query	bool								false	"Flag to include total record count in the response (default: false)"
//	@Success		200			{object}	response.PostManagementByOfficeAndPostAPIResponse	"Post management details retrieved successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401			{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403			{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404			{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409			{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500			{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/posts [get]
func (pmh *PostManagementHandler) PostManagementByOfficeAndPostHandler(ctx *gin.Context) {
	var req PostManagementQueryRequest

	// Bind the query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementQueryRequest: %s", err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementQueryRequest: %s", err)
		return
	}
	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}

	var postList []domain.PostManagementMaster
	var err error

	// Case 1: If both post-id and office-id are provided, fetch by post-id and office-id
	if req.PostID != 0 {
		postList, err = pmh.svc.PostManagementByOfficeIDQueryMDW(ctx, req.OfficeID, req.PostID, req.MetaDataRequest)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "PostManagementByOfficeIDQueryMDW Repo call failed: %s", err.Error())
			return
		}
		// Case 3: If only cadre-name and office-id are provided, fetch by cadre-name and office-id
	} else if req.CadreName != "" {
		postList, err = pmh.svc.PostManagementByCadreAndOfficeQuery(ctx, req.CadreName, req.OfficeID, req.MetaDataRequest)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "PostManagementByCadreAndOfficeQuery Repo call failed: %s", err.Error())
			return
		}
		// Case 4: If only office-id is provided, fetch by office-id
	} else {
		postList, err = pmh.svc.PostManagementByOfficeIDQuery(ctx, req.OfficeID)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "PostManagementByOfficeIDQuery Repo call failed: %s", err.Error())
			return
		}
	}

	rsp := response.NewPostManagementByOfficeAndPostResponse(postList)
	metadata := port.NewMetaDataResponse(req.MetaDataRequest.Skip, req.MetaDataRequest.Limit, len(rsp))
	apiRsp := response.PostManagementByOfficeAndPostAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementByOfficeAndPostHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type DesignationMasterCombinedResponse struct {
	DesignationID   int    `json:"designation_id"`
	DesignationName string `json:"designation_name"`
	Designation     string `json:"designation"`
	GroupName       string `json:"group_name"`
	CadreName       string `json:"cadre_name"`
	CadreId         int    `json:"cadre_id"`
	GroupId         int16  `json:"group_id"`
}

type PostManagementFilledStatusRequest struct {
	FilledStatus string `form:"filled-status" validate:"required,oneof=Vacant or Filled"`
	PostId       int    `form:"post-id" validate:"required"`
	UpdatedDate  string `form:"updated-date" validate:"required"`
}

// PostManagementChangFilledStatusHandler godoc
//
//		@Summary		Change Filled Status by Post ID
//		@Description	Changes the filled status of a post using the specified Post ID.
//		@Tags			Post Management
//		@Accept			json
//		@Produce		json
//		@Param			filled-status		query	string	true	"New filled status of the post (Vacant or Filled)"
//		@Param			post-id				query	int		true	"ID of the post to change the filled status"
//	 	@Param 			updated-date		query	string	true	"Date on which the filled status is updated"
//		@Success		200	{object}	response.PostManagementChangFilledStatusAPIResponse	"Filled status changed successfully"
//		@Failure		400	{object}	apierrors.APIErrorResponse										"Validation error"
//		@Failure		401	{object}	apierrors.APIErrorResponse										"Unauthorized error"
//		@Failure		403	{object}	apierrors.APIErrorResponse										"Forbidden error"
//		@Failure		404	{object}	apierrors.APIErrorResponse										"Post not found"
//		@Failure		500	{object}	apierrors.APIErrorResponse										"Internal server error"
//		@Router			/post-management/posts/filled-status [put]
func (pmh *PostManagementHandler) PostManagementChangFilledStatusHandler(ctx *gin.Context) {
	var req PostManagementFilledStatusRequest

	// Bind URI parameters
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementMasterChangeRequest: %s", err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementMasterChangeRequest: %s", err)
		return
	}

	// Call the service to change the filled status by PostID
	message, err := pmh.svc.PostManagementChangeFilledStatus(ctx, req.FilledStatus, req.PostId, req.UpdatedDate)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementChangFilledStatusByPostID Repo call failed: %s", err.Error())
		return
	}

	// Create the response data
	rsp := response.NewPostManagementChangFilledStatusResponse(message)
	apiRsp := response.PostManagementChangFilledStatusAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementChangFilledStatusByPostIDHandler response: %v", apiRsp)
	// Send the success response
	handleSuccess(ctx, apiRsp)
}

type getListPostManagementRequest struct {
	DivisionOfficeId int `form:"division-office-id" validate:"required" example:"1234"`
	port.MetaDataRequest
}

// ListPostManagementHandler godoc
//
//	@Summary		List Post Management by Division Office
//	@Description	Retrieves a list of post management data filtered by Division Office ID.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			division-office-id	query	int		true	"ID of the Division Office"
//	@Param			skip				query	int		false	"Number of records to skip"
//	@Param			limit				query	int		false	"Number of records to fetch"
//	@Success		200	{object}	response.FetchPostManagementresponse						"List retrieved successfully"
//	@Failure		400	{object}	apierrors.APIErrorResponse										"Validation error"
//	@Failure		401	{object}	apierrors.APIErrorResponse										"Unauthorized error"
//	@Failure		403	{object}	apierrors.APIErrorResponse										"Forbidden error"
//	@Failure		404	{object}	apierrors.APIErrorResponse										"Division office not found"
//	@Failure		500	{object}	apierrors.APIErrorResponse										"Internal server error"
//	@Router			/post-management/posts/division-office [get]
func (ch *PostManagementHandler) ListPostManagementHandler(ctx *gin.Context) {
	var req getListPostManagementRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for getListPostManagementRequest: %s", err.Error())
		return
	}

	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for getListPostManagementRequest: %s", err.Error())
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for getListPostManagementRequest: %s", err.Error())
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	postmgmt, err := ch.svc.ListPostManagement(ctx, req.DivisionOfficeId, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "ListPostManagement Repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	log.Debug(ctx, "ListPostManagement response", postmgmt)
	rsp := response.NewPostManagementMaker(postmgmt)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.FetchPostManagementresponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	handleSuccess(ctx, apiRsp)
}

type getListAvailablePostsRequest struct {
	OfficeId int `form:"office-id" validate:"required"`
	port.MetaDataRequest
}

// ListAvailablePostsHandler godoc
//
// @Summary      List available posts
// @Description  Retrieves a paginated list of available posts for a given office ID.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        office-id  query     int  true   "Office ID"
// @Param        limit      query     int  false  "Number of records to fetch (default: unlimited)"
// @Param        skip       query     int  false  "Number of records to skip (for pagination)"
// @Success      200  {object}  response.ListAvailablePostsresponse  "Successful response with available posts"
// @Failure      400  {object}  apierrors.APIErrorResponse           "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse           "Internal server error"
// @Router       /posts/available-posts [get]
func (ch *PostManagementHandler) ListAvailablePostsHandler(ctx *gin.Context) {
	var req getListAvailablePostsRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for getListAvailablePostsRequest: %s", err.Error())
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for getListAvailablePostsRequest: %s", err.Error())
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	postmgmt, err := ch.svc.ListAvailablePosts(ctx, req.OfficeId, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "ListAvailablePosts Repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	log.Debug(ctx, "ListAvailablePosts response", postmgmt)
	rsp := response.NewListAvailablePostsMaker(postmgmt)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListAvailablePostsresponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	handleSuccess(ctx, apiRsp)
}

// ListVacantPostsHandler godoc
//
// @Summary      List vacant posts
// @Description  Retrieves a paginated list of vacant posts for a given office ID.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        office-id  query     int  true   "Office ID"
// @Param        limit      query     int  false  "Number of records to fetch (default: unlimited)"
// @Param        skip       query     int  false  "Number of records to skip (for pagination)"
// @Success      200  {object}  response.ListVacantPostsresponse  "Successful response with vacant posts"
// @Failure      400  {object}  apierrors.APIErrorResponse         "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse         "Internal server error"
// @Router       /posts/vacant-posts [get]
func (ch *PostManagementHandler) ListVacantPostsHandler(ctx *gin.Context) {
	var req getListAvailablePostsRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for getListVacantPostsRequest: %s", err.Error())
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for getListVacantPostsRequest: %s", err.Error())
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	postmgmt, err := ch.svc.ListVacantPosts(ctx, req.OfficeId, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "ListVacantPosts Repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	log.Debug(ctx, "ListVacantPosts response", postmgmt)
	rsp := response.NewListVacantPostsMaker(postmgmt)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListVacantPostsresponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	handleSuccess(ctx, apiRsp)
}

type GroupMasterRequest struct {
	port.MetaDataRequest
}

// ListGroupMasterHandler lists group master entries.
//
// @Summary List Group Master
// @Description Returns a list of group master entries with pagination and metadata
// @Tags PostManagement
// @Accept json
// @Produce json
// @Param Limit query int false "Number of records to return" minimum(1)
// @Param Skip query int false "Number of records to skip for pagination" minimum(0)
// @Param Sort query string false "Sort order (e.g. name asc, id desc)"
// @Success 200 {object} response.ListGroupAPIResponse
// @Failure 400 {object} apierrors.APIErrorResponse
// @Failure 500 {object} apierrors.APIErrorResponse
// @Router /posts/group-master [get]
func (ch *PostManagementHandler) ListGroupMasterHandler(ctx *gin.Context) {
	var req GroupMasterRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	groupMaster, err := ch.svc.ListGroupMaster(ctx, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "ListGroupMaster Repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	log.Debug(ctx, "ListGroupMaster response", groupMaster)
	rsp := response.NewListGroupMasterMaker(groupMaster)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListGroupAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	handleSuccess(ctx, apiRsp)
}

type OfficeDetailsRequest struct {
	OfficeId int64 `form:"office-id" validate:"required"`
	port.MetaDataRequest
}

// ListOfficeDetailsHandler returns a paginated list of office details.
//
// @Summary List Office Details
// @Description Get paginated list of office details by OfficeId
// @Tags PostManagement
// @Accept json
// @Produce json
// @Param office-id query int true "Office ID"
// @Param Limit query int false "Number of records to return" minimum(1)
// @Param Skip query int false "Number of records to skip for pagination" minimum(0)
// @Param Sort query string false "Sorting (e.g. name asc, id desc)"
// @Success 200 {object} response.ListOfficeDetailsAPIResponse
// @Failure 400 {object} apierrors.APIErrorResponse
// @Failure 500 {object} apierrors.APIErrorResponse
// @Router /posts/office-details [get]
func (ch *PostManagementHandler) ListOfficeDetailsHandler(ctx *gin.Context) {
	var req OfficeDetailsRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	officeDetails, err := ch.svc.ListOfficeDetails(ctx, req.OfficeId, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "ListOfficeDetails Repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	log.Debug(ctx, "ListOfficeDetails response", officeDetails)
	rsp := response.NewListOfficeDetailsMaker(officeDetails)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListOfficeDetailsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	handleSuccess(ctx, apiRsp)
}

type ListGroupCadreRequest struct {
	GroupID int64 `form:"group-id" validate:"required"`
	port.MetaDataRequest
}

// ListGroupCadreHandler returns a paginated list of group cadre details.
//
// @Summary List Group Cadre Details
// @Description Get paginated list of cadre details for a given group
// @Tags PostManagement
// @Accept json
// @Produce json
// @Param group-id query int true "Group ID"
// @Param Limit query int false "Number of records to return" minimum(1)
// @Param Skip query int false "Number of records to skip for pagination" minimum(0)
// @Param Sort query string false "Sorting (e.g. name asc, id desc)"
// @Success 200 {object} response.ListGroupCadreAPIResponse
// @Failure 400 {object} apierrors.APIErrorResponse
// @Failure 500 {object} apierrors.APIErrorResponse
// @Router /posts/group-cadre [get]
func (ch *PostManagementHandler) ListGroupCadreHandler(ctx *gin.Context) {
	var req ListGroupCadreRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	groupCadreDetails, err := ch.svc.ListGroupCadre(ctx, req.GroupID, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "ListGroupCadre Repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	log.Debug(ctx, "ListGroupCadre response", groupCadreDetails)
	rsp := response.NewListGroupCadreMaker(groupCadreDetails)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListGroupCadreAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	handleSuccess(ctx, apiRsp)
}

// CreatePostHandler godoc
//
// @Summary      Bulk create posts
// @Description  Creates one or more posts under a given office. Supports bulk creation by specifying `number_of_posts`.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        request  body      domain.CreatePostRequest  true  "Post creation request"
// @Success      200      {object}  port.APIResponse   "Successfully created post(s)"
// @Failure      400      {object}  apierrors.APIErrorResponse "Validation or binding error"
// @Failure      500      {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /posts/bulk/create [post]
func (ch *PostManagementHandler) CreatePostHandler(ctx *gin.Context) {
	var req domain.CreatePostRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		return
	}

	postIDs, err := ch.svc.CreatePost(ctx, &req)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	data := gin.H{"postmanagement_ids": postIDs}
	apiRsp := port.APIResponse{
		StatusCodeAndMessage: port.StatusCodeAndMessage{
			StatusCode: 200,
			Success:    true,
			Message:    fmt.Sprintf("%d post(s) created successfully", len(postIDs)),
		},
		Data: data,
	}
	handleSuccess(ctx, apiRsp)
}

// UpdatePostHandler updates a post's office and administrative details.
//
// @Summary      Update Post Management Master
// @Description  Updates details like office name, type, and associated post for a given office ID.
// @Tags         PostManagement
// @Accept       json
// @Produce      json
// @Param        request  body  domain.UpdatePostRequest  true  "Post update payload"
// @Success      200      {object}  response.UpdatePostManagementResponse  "Post updated successfully"
// @Failure      400      {object}  apierrors.APIErrorResponse  "Validation error"
// @Failure      401      {object}  apierrors.APIErrorResponse  "Unauthorized"
// @Failure      403      {object}  apierrors.APIErrorResponse  "Forbidden"
// @Failure      404      {object}  apierrors.APIErrorResponse  "Post not found"
// @Failure      500      {object}  apierrors.APIErrorResponse  "Internal server error"
// @Router       /posts/update-pmm [put]
func (ch *PostManagementHandler) UpdatePostHandler(ctx *gin.Context) {
	var req domain.UpdatePostRequest
	if err := apiutility.BindAndValidate(ctx, &req, false, false, true); err != nil {
		log.Error(ctx, err)
		return
	}

	resp, err := ch.svc.UpdatePost(ctx, &req)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.UpdatePostManagementResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 resp,
	}

	handleSuccess(ctx, apiRsp)
}

type PostManagementByOfficeIDRequest struct {
	OfficeID int64 `form:"office-id" validate:"required"`
	port.MetaDataRequest
}

// PostManagementByOfficeIDHandler godoc
//
// @Summary      Get Post Management details by Office ID
// @Description  Fetch all vacant posts in a given office. If no posts or no vacant posts are found, a message will be returned.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        office-id   query     int     true   "Office ID"
// @Param        skip        query     int     false  "Number of records to skip (for pagination)"
// @Param        limit       query     int     false  "Number of records to return (for pagination)"
// @Success      200  {object}  response.PostManagementByOfficeIDResponse1  "Posts fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse                  "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse                  "Internal server error"
// @Router       /office-post-details [get]
func (ch *PostManagementHandler) PostManagementByOfficeIDHandler(ctx *gin.Context) {
	var req PostManagementByOfficeIDRequest
	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	// Fetch the list of posts
	postList, err := ch.svc.PostManagementByOfficeID(ctx, req.OfficeID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementByOfficeID Repo call failed: %s", err.Error())
		return
	}

	// If there are no posts found
	if len(postList) == 0 {
		handleSuccess(ctx, response.PostManagementByOfficeIDResponse{
			StatusCodeAndMessage: port.ListSuccess,
			Data:                 "no post vacant in this office",
		})
		return
	}

	// Fetch vacant posts from the list
	vacantPost, err := ch.svc.PostManagementByOfficeIDVacant(ctx, postList, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementByOfficeIDVacant Repo call failed: %s", err.Error())
		return
	}

	// If no vacant posts are found
	if len(vacantPost) == 0 {
		handleSuccess(ctx, response.PostManagementByOfficeIDResponse{
			StatusCodeAndMessage: port.ListSuccess,
			Data:                 "no post vacant in this office",
		})
		return
	}

	// Prepare the successful response
	rsp := response.NewPostManagementByOfficeIDResponse(vacantPost)
	metadata := port.NewMetaDataResponse(req.MetaDataRequest.Skip, req.MetaDataRequest.Limit, len(rsp))
	apiRsp := response.PostManagementByOfficeIDResponse1{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	handleSuccess(ctx, apiRsp)
}

type PostDetailsRequest struct {
	PostID int64 `form:"post-id" validate:"required"`
	port.MetaDataRequest
}

// GetPostDetailsHandler godoc
//
// @Summary      Get post details
// @Description  Fetch detailed information of a post by post-id with pagination support.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        post-id   query     int     true   "Post ID"
// @Param        skip      query     int     false  "Number of records to skip (for pagination)"
// @Param        limit     query     int     false  "Number of records to return (for pagination)"
// @Success      200  {object}  response.PostDetailsAPIResponse   "Post details fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse        "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse        "Internal server error"
// @Router       /posts/post-details [get]
func (h *PostManagementHandler) GetPostDetailsHandler(ctx *gin.Context) {
	var req PostDetailsRequest
	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	results, err := h.svc.GetPostDetailsByPostID(ctx, req.PostID, int64(req.Skip), int64(req.Limit))
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "GetPostDetailsByPostID Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewPostDetailsResponse(results)
	metadata := port.NewMetaDataResponse(req.MetaDataRequest.Skip, req.MetaDataRequest.Limit, len(rsp))
	apiRsp := response.PostDetailsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	handleSuccess(ctx, apiRsp)
}

// GetCLGrantingPostsHandler godoc
//
// @Summary      Get CL Granting posts
// @Description  Fetch posts that are authorized for CL granting by post-id with pagination support.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        post-id   query     int     true   "Post ID"
// @Param        skip      query     int     false  "Number of records to skip (for pagination)"
// @Param        limit     query     int     false  "Number of records to return (for pagination)"
// @Success      200  {object}  response.PostDetailsAPIResponse   "CL granting posts fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse        "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse        "Internal server error"
// @Router       /posts/cl-granting-posts [get]
func (h *PostManagementHandler) GetCLGrantingPostsHandler(ctx *gin.Context) {
	var req PostDetailsRequest
	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	results, err := h.svc.GetCLGrantingPosts(ctx, req.PostID, int64(req.Skip), int64(req.Limit))
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "GetCLGrantingPosts Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewPostDetailsResponse(results)
	metadata := port.NewMetaDataResponse(req.MetaDataRequest.Skip, req.MetaDataRequest.Limit, len(rsp))
	apiRsp := response.PostDetailsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	handleSuccess(ctx, apiRsp)
}

// GetDDOFilteredPostsHandler godoc
//
// @Summary      Get DDO Filtered posts
// @Description  Fetch posts filtered by DDO using post-id with pagination support.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        post-id   query     int     true   "Post ID"
// @Param        skip      query     int     false  "Number of records to skip (for pagination)"
// @Param        limit     query     int     false  "Number of records to return (for pagination)"
// @Success      200  {object}  response.PostDetailsAPIResponse   "DDO filtered posts fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse        "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse        "Internal server error"
// @Router       /posts/ddo-filtered-posts [get]
func (h *PostManagementHandler) GetDDOFilteredPostsHandler(ctx *gin.Context) {
	var req PostDetailsRequest
	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	results, err := h.svc.GetDDOFilteredPosts(ctx, req.PostID, int64(req.Skip), int64(req.Limit))
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "GetDDOFilteredPosts Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewPostDetailsResponse(results)
	metadata := port.NewMetaDataResponse(req.MetaDataRequest.Skip, req.MetaDataRequest.Limit, len(rsp))
	apiRsp := response.PostDetailsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	handleSuccess(ctx, apiRsp)
}

// FetchPostsByOfficeIDHandler2 godoc
//
// @Summary      Fetch posts by office ID
// @Description  Retrieves post management data for a given office ID.
// @Description  - If `post-id` is provided, fetches maker details for that post.
// @Description  - If `vacant=true`, fetches vacant active posts.
// @Description  - If `cadre-id` is provided, fetches posts by office cadre.
// @Description  - Otherwise, fetches all active posts for the office.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        office-id   path      int     true   "Office ID"
// @Param        post-id     query     int     false  "Filter by specific Post ID"
// @Param        cadre-id    query     int  false  "Filter by Cadre ID"
// @Param        vacant      query     bool    false  "Fetch only vacant posts (true/false)"
// @Success      200         {object}  response.FetchPostsByOfficeIDAPIResponse2  "Successful response with posts"
// @Failure      400         {object}  apierrors.APIErrorResponse  "Validation or binding error"
// @Failure      500         {object}  apierrors.APIErrorResponse  "Internal server error"
// @Router       /post-management/office-post-details/{office-id}/posts-summary [get]
func (pmh *PostManagementHandler) FetchPostsByOfficeIDHandler2(ctx *gin.Context) {
	var req PostManagementListRequest123
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementListRequest123: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementListRequest123: %s", err)
		return
	}

	postIDStr := ctx.Query("post-id")
	cadreID := ctx.Query("cadre-id")
	isVacant := ctx.DefaultQuery("vacant", "false")

	var masterList []domain.PostManagementMasterNew
	var makerList []domain.PostManagementMaker
	var err error

	if postIDStr != "" {
		postID, err := strconv.ParseInt(postIDStr, 10, 64)
		if err != nil {
			apierrors.HandleErrorWithStatusCodeAndMessage(ctx, apierrors.AppErrorValidationError, "invalid post-id", err)
			return
		}
		makerList, err = pmh.svc.PostManagementByOfficeIDMDWMaker(ctx, req.OfficeID, postID)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "PostManagementByOfficeIDMDWMaker Repo call failed: %s", err.Error())
			return
		}
	} else if isVacant == "true" {
		masterList, err = pmh.svc.FetchVacantActivePostByOfficeID2(ctx, req.OfficeID)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "FetchVacantActivePostByOfficeID Repo call failed: %s", err.Error())
			return
		}
	} else if cadreID != "" {
		masterList, err = pmh.svc.FetchAllActivePostByOfficeCadre(ctx, req.OfficeID, cadreID)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "FetchAllActivePostByOfficeID Repo call failed: %s", err.Error())
			return
		}
	} else {
		masterList, err = pmh.svc.FetchAllActivePostByOfficeID2(ctx, req.OfficeID)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "FetchAllActivePostByOfficeID Repo call failed: %s", err.Error())
			return
		}
	}

	rsp := response.NewFetchPostsByOfficeIDResponse2(masterList, makerList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.FetchPostsByOfficeIDAPIResponse2{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "FetchPostsByOfficeIDHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// func (pph *PosttoPostMappingrHandler) GetPostAndEmployeeHierarchy(ctx *gin.Context) {
// 	var req struct {
// 		OfficeID int `json:"office_id" binding:"required"`
// 	}

// 	// Bind JSON body
// 	if err := ctx.ShouldBindJSON(&req); err != nil {
// 		apierrors.HandleBindingError(ctx, err)
// 		return
// 	}

// 	// Call service to get results
// 	posts, err := pph.svc.GetPostAndEmployeeHierarchy(ctx, req.OfficeID)
// 	if err != nil {
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}

// 	// Send raw JSON array directly (no response struct)
// 	ctx.JSON(http.StatusOK, map[string]interface{}{
// 		"status_code":            200,
// 		"success":                true,
// 		"message":                "list retrieved successfully",
// 		"skip":                   0,
// 		"limit":                  1000,
// 		"returned_records_count": len(posts),
// 		"data":                   posts,
// 	})
// }

// SavePostToPostMappings godoc
//
// @Summary      Save Post-to-Post Mappings
// @Description  Saves or updates multiple post-to-post mappings with metadata about the operation.
// @Tags         Post-to-Post Mapping
// @Accept       json
// @Produce      json
// @Param        payload  body  domain.PostMappingPayload  true  "Post-to-Post Mapping Payload"
// @Success      200  {object}  map[string]interface{}  "Mappings saved successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request payload"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /ptop-mappings-makers/save [post]
func (h *PosttoPostMappingrHandler) SavePostToPostMappings(ctx *gin.Context) {
	var payload domain.PostMappingPayload
	if err := ctx.ShouldBindJSON(&payload); err != nil {
		apierrors.HandleBindingError(ctx, err)
		return
	}
	// fmt.Println(payload)

	err := h.svc.SavePostMappingsRepo(ctx, payload)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	ctx.JSON(http.StatusOK, map[string]interface{}{
		"status_code": 200,
		"success":     true,
		"message":     "Data Updated successfully",
	})
}

// IdentifyHeadOfOffice godoc
//
// @Summary      Identify Head of Office
// @Description  Identify the Head of Office for a given office ID.
// @Tags         Post-to-Post Mapping
// @Accept       json
// @Produce      json
// @Param        request   body    domain.HeadOfOfficeRequest  true  "Office ID payload"
// @Success      200  {object}  domain.HeadOfOfficeResponse   "Head of Office identified successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse    "Invalid request payload"
// @Failure      500  {object}  apierrors.APIErrorResponse    "Internal server error"
// @Router       /ptop-mappings/hoo [post]
func (rmh *PosttoPostMappingrHandler) IdentifyHeadOfOffice(ctx *gin.Context) {
	var req domain.HeadOfOfficeRequest

	if err := ctx.ShouldBindJSON(&req); err != nil {
		log.Error(ctx, "Binding failed for HeadOfOfficeRequest", err.Error())
		apierrors.HandleBindingError(ctx, err)
		return
	}

	result, err := rmh.svc.IdentifyHeadOfOffice(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Failed to identify Head of Office", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := domain.HeadOfOfficeResponse{
		StatusCode: 200,
		Message:    "Head of Office identified successfully",
		Data:       result,
	}

	log.Debug(ctx, "IdentifyHeadOfOffice API Response: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// UpdateHeadOfOffice godoc
//
// @Summary      Update Head of Office
// @Description  Update the Head of Office for a given office ID. The caller's post ID is validated for authorisation.
// @Tags         Post-to-Post Mapping
// @Accept       json
// @Produce      json
// @Param        request   body      domain.UpdateHeadOfOfficeRequest  true  "Update Head of Office payload"
// @Success      200  {object}  domain.GenericSuccessResponse   "Head of Office updated successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse      "Invalid request payload"
// @Failure      401  {object}  apierrors.APIErrorResponse      "Unauthorized (requestor post not allowed to update)"
// @Failure      404  {object}  apierrors.APIErrorResponse      "Office/Post not found"
// @Failure      500  {object}  apierrors.APIErrorResponse      "Internal server error"
// @Router       /ptop-mappings/head-office/update [post]
func (rmh *PosttoPostMappingrHandler) UpdateHeadOfOffice(ctx *gin.Context) {
	var req domain.UpdateHeadOfOfficeRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		log.Error(ctx, "Binding failed for UpdateHeadOfOfficeRequest", err.Error())
		apierrors.HandleBindingError(ctx, err)
		return
	}

	// Authorisation + update in one repo call
	err := rmh.svc.UpdateHeadOfOffice(ctx,
		req.OfficeID,
		req.PostID,
		req.RequestorPostID)

	if err != nil {
		log.Error(ctx, "Failed to update Head of Office", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := domain.GenericSuccessResponse{
		StatusCode: 200,
		Message:    "Head of Office updated successfully",
	}
	log.Debug(ctx, "UpdateHeadOfOffice API Response: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// GetHeadPostOccupant godoc
//
// @Summary      Get Head Post Occupant
// @Description  Fetch the occupant details of the Head of Office post for a given office ID.
// @Tags         Post-to-Post Mapping
// @Accept       json
// @Produce      json
// @Param        office_id   query     int64   true   "Office ID"
// @Success      200  {object}  map[string]interface{}  "Head post occupant fetched successfully"
// @Failure      400  {object}  map[string]interface{}  "Missing or invalid office_id"
// @Failure      404  {object}  apierrors.APIErrorResponse "Head of Office occupant not found"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /ptop-mappings/head-office/search [get]
func (rmh *PosttoPostMappingrHandler) GetHeadPostOccupant(ctx *gin.Context) {
	officeIDStr := ctx.Query("office_id")
	if officeIDStr == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"status_code": 400,
			"message":     "Missing required query param: office_id",
		})
		return
	}

	officeID, err := strconv.ParseInt(officeIDStr, 10, 64)
	if err != nil || officeID <= 0 {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"status_code": 400,
			"message":     "Invalid office_id",
		})
		return
	}

	result, err := rmh.svc.GetHeadPostOccupant(ctx, officeID)
	if err != nil {
		log.Error(ctx, "Failed to fetch head post occupant: ", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"status_code": 200,
		"message":     "Head post occupant fetched successfully",
		"data":        result,
	})
}

// GetPostDetailsForHOO godoc
//
// @Summary      Get Post Details for Head of Office
// @Description  Fetch the list of post details eligible for Head of Office in a given office.
// @Tags         Post-to-Post Mapping
// @Accept       json
// @Produce      json
// @Param        office_id   query     int64   true   "Office ID"
// @Success      200  {object}  map[string]interface{}  "Head of Office post details fetched successfully"
// @Failure      400  {object}  map[string]interface{}  "Missing or invalid office_id"
// @Failure      404  {object}  apierrors.APIErrorResponse "Head of Office post details not found"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /ptop-mappings/post/search [get]
func (rmh *PosttoPostMappingrHandler) GetPostDetailsForHOO(ctx *gin.Context) {
	officeIDStr := ctx.Query("office_id")
	if officeIDStr == "" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"status_code": 400,
			"message":     "Missing required query param: office_id",
		})
		return
	}

	officeID, err := strconv.ParseInt(officeIDStr, 10, 64)
	if err != nil || officeID <= 0 {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"status_code": 400,
			"message":     "Invalid office_id",
		})
		return
	}

	result, err := rmh.svc.GetPostDetailsForHOO(ctx, officeID)
	if err != nil {
		log.Error(ctx, "Failed to fetch head post occupant: ", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	ctx.JSON(http.StatusOK, gin.H{
		"status_code": 200,
		"message":     "Head post occupant fetched successfully",
		"data":        result,
	})
}

// FetchAllPostsByOfficeIDHandler2 godoc
//
// @Summary      Fetch posts by office ID
// @Description  Retrieves post management data for a given office ID.
// @Description  - If `post-id` is provided, fetches maker details for that post.
// @Description  - If `vacant=true`, fetches vacant active posts.
// @Description  - If `cadre-id` is provided, fetches posts by office cadre.
// @Description  - Otherwise, fetches all active posts for the office.
// @Tags         Post Management
// @Accept       json
// @Produce      json
// @Param        office-id   path      int     true   "Office ID"
// @Param        post-id     query     int     false  "Filter by specific Post ID"
// @Param        cadre-id    query     int  false  "Filter by Cadre ID"
// @Param        vacant      query     bool    false  "Fetch only vacant posts (true/false)"
// @Success      200         {object}  response.FetchPostsByOfficeIDAPIResponse2  "Successful response with posts"
// @Failure      400         {object}  apierrors.APIErrorResponse  "Validation or binding error"
// @Failure      500         {object}  apierrors.APIErrorResponse  "Internal server error"
// @Router       /post-management/office-post-details/{office-id}/posts-summary-all [get]
func (pmh *PostManagementHandler) FetchAllPostsByOfficeIDHandler2(ctx *gin.Context) {
	var req PostManagementListRequest
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementListRequest123: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementListRequest123: %s", err)
		return
	}

	var masterList []domain.PostManagementMasterNew
	//var makerList []domain.PostManagementMaker
	var err error

	masterList, err = pmh.svc.FetchAllPostByOfficeID2(ctx, req.OfficeID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "FetchAllActivePostByOfficeID Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewFetchPostsByOfficeIDAllResponse2(masterList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.FetchPostsByOfficeIDAllAPIResponse2{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "FetchAllPostsByOfficeIDHandler2 resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type PostManagementListRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

type PostManagementD1 struct {
	ValidFrom time.Time `form:"valid-from" time_format:"2006-01-02" validate:"required"`
	ValidTo   time.Time `form:"valid-to" time_format:"2006-01-02" validate:"required,gtfield=ValidFrom"`
	port.MetaDataRequest
}

func (pmh *PostManagementHandler) PostManagementByOfficeAndPostHandlerD1(ctx *gin.Context) {
	var req PostManagementD1

	// Bind the query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementQueryRequest: %s", err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementQueryRequest: %s", err)
		return
	}
	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}

	var postList []domain.PostManagementMaster
	var err error

	postList, err = pmh.svc.PostManagementByOfficeIDQueryD1(ctx, req.ValidFrom, req.ValidTo, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementByOfficeIDQuery Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewPostManagementByOfficeAndPostResponse(postList)
	metadata := port.NewMetaDataResponse(req.MetaDataRequest.Skip, req.MetaDataRequest.Limit, len(rsp))
	apiRsp := response.PostManagementByOfficeAndPostAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementByOfficeAndPostHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostDetailRequest struct {
	OfficeID int64 `uri:"office-id" validate:"required"`
	port.MetaDataRequest
}

func (h *PostManagementHandler) GetPostDetailHandler(ctx *gin.Context) {
	var req GetPostDetailRequest
	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}

	results, err := h.svc.GetPostDetailByOfficeID(ctx, req.OfficeID, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "GetPostDetailByOfficeID Repo call failed: %s", err.Error())
		return
	}
	rsp := response.NewPostDetails1(results)

	metadata := port.NewMetaDataResponse(req.MetaDataRequest.Skip, req.MetaDataRequest.Limit, len(rsp))
	apiRsp := response.GetPostDetails1APIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetPostDetailHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}
