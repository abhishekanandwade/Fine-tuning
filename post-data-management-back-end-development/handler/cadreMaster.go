package handler

import (
	"math"
	"pmdm/core/domain"
	"pmdm/core/port"
	"pmdm/handler/response"
	repo "pmdm/repo/postgres"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/volatiletech/null/v9"
	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
	validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
)

// CadreMasterHandler represents the HTTP handler for cadre master-related requests
type CadreMasterHandler struct {
	svc *repo.CadreMasterRepository
}

// NewCadreMasterHandler creates a new CadreMasterHandler instance
func NewCadreMasterHandler(svc *repo.CadreMasterRepository) *CadreMasterHandler {
	return &CadreMasterHandler{
		svc,
	}
}

type CadreMasterListRequest struct {
	GroupCode string `form:"group-code" validate:"omitempty"`
	port.MetaDataRequest
}

// ListCadres godoc
//
//	@Summary		List Cadre Details
//	@Description	List all cadre types or filter by group-code if provided
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			group-code	query		string										false	"Filter cadres by group code"
//	@Param       metaDataRequest	query		port.MetaDataRequest	false  		"Metadata request containing skip, limit, orderBy & sortBy"
//	@Success		200			{object}	response.ListCadresAPIResponse				"Cadre list retrieved successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401			{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403			{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404			{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409			{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500			{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/cadres [get]
func (cmh *CadreMasterHandler) ListCadresHandler(ctx *gin.Context) {
	var req CadreMasterListRequest

	// Bind query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for CadreMasterListRequest: %s", err.Error())
		return
	}

	// Check if GroupCode is provided
	if req.GroupCode != "" {
		if err := validation.ValidateStruct(req); err != nil {
			apierrors.HandleValidationError(ctx, err)
			log.Error(ctx, "Validation failed for CadreMasterListRequest: %s", err.Error())
			return
		}

		if req.Limit == 0 {
			req.Limit = math.MaxInt32
		}

		cadreList, err := cmh.svc.CadreMasterByGroupCodeQuery(ctx, req.GroupCode, req.MetaDataRequest)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			log.Error(ctx, "CadreMasterByGroupCodeQuery Repo call failed: %s", err.Error())
			return
		}

		rsp := response.NewListCadresResponse(cadreList)
		metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
		apiRsp := response.ListCadresAPIResponse{
			StatusCodeAndMessage: port.ListSuccess,
			MetaDataResponse:     metadata,
			Data:                 rsp,
		}

		log.Debug(ctx, "ListCadresHandler response: %v", apiRsp)
		handleSuccess(ctx, apiRsp)

	} else {
		if err := validation.ValidateStruct(req); err != nil {
			apierrors.HandleValidationError(ctx, err)
			log.Error(ctx, "Validation failed for CadreMasterListRequest: %s", err.Error())
			return
		}
		cadreList, err := cmh.svc.ListCadresQry(ctx, req.MetaDataRequest)
		if err != nil {
			apierrors.HandleDBError(ctx, err)
			return
		}
		rsp := response.NewListCadresResponse(cadreList)
		metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
		apiRsp := response.ListCadresAPIResponse{
			StatusCodeAndMessage: port.ListSuccess,
			MetaDataResponse:     metadata,
			Data:                 rsp,
		}
		log.Debug(ctx, "ListCadresHandler response: %v", apiRsp)
		handleSuccess(ctx, apiRsp)
	}
}

type CadreMasterListAllRequest struct {
	port.MetaDataRequest
}

// ListAllCadres godoc
//
//	@Summary		List All Cadre Details
//	@Description	List all cadre types or filter by group-code if provided
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			group-code	query		string										false	"Filter cadres by group code"
//	@Param       metaDataRequest	query		port.MetaDataRequest	false  		"Metadata request containing skip, limit, orderBy & sortBy"
//	@Success		200			{object}	response.ListCadresAPIResponse				"Cadre list retrieved successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401			{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403			{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404			{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409			{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500			{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/cadres/all-cadres [get]
func (cmh *CadreMasterHandler) ListAllCadresHandler(ctx *gin.Context) {
	var req CadreMasterListAllRequest

	// Bind query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for CadreMasterListAllRequest: %s", err.Error())
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for CadreMasterListAllRequest: %s", err.Error())
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	cadreList, err := cmh.svc.ListAllCadresQry(ctx, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}
	rsp := response.NewListAllCadresResponse(cadreList)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListAllCadresAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	log.Debug(ctx, "ListAllCadresHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)

}

type CreateCadreRequest struct {
	CadreName string    `json:"cadre_name" validate:"required"`
	GroupName string    `json:"group_name" validate:"required"`
	PayLevel  int32     `json:"pay_level" validate:"required"`
	GradePay  int32     `json:"grade_pay" validate:"required"`
	CreatedBy string    `json:"created_by" validate:"required"`
	ValidFrom time.Time `json:"valid_from" validate:"required"`
	ValidTo   time.Time `json:"valid_to" validate:"required"`
	Status    string    `json:"status" validate:"required"`
	Remarks   string    `json:"remarks" validate:"required"`
	GroupCode int16     `json:"group_id" validate:"required"`
}

// CreateCadreMaster godoc
//
//	@Summary        Create Cadre Master
//	@Description    Creates a post cadre master entry.
//	@Tags           Post Management
//	@Accept         json
//	@Produce        json
//	@Param          CreateCadreRequest   body   CreateCadreRequest  true    "cadre details to create"
//	@Success        200 {object} response.CreateCadreMasterAPIResponse "Cadre master details created successfully"
//	@Failure        400 {object} apierrors.APIErrorResponse "Validation error"
//	@Failure        401 {object} apierrors.APIErrorResponse "Unauthorized access"
//	@Failure        403 {object} apierrors.APIErrorResponse "Forbidden access"
//	@Failure        404 {object} apierrors.APIErrorResponse "Resource not found"
//	@Failure        409 {object} apierrors.APIErrorResponse "Conflict in the data"
//	@Failure        500 {object} apierrors.APIErrorResponse "Internal server error"
//	@Router         /post-management/cadres/create [post]
func (cmh *CadreMasterHandler) CreateCadreMasterHandler(ctx *gin.Context) {
	var req CreateCadreRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementCreateRequests: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementFetchRequest: %s", err)
		return
	}

	createCadre := domain.CadreMaster{

		CadreName: null.StringFrom(req.CadreName),
		PayLevel:  null.Int32From(int32(req.PayLevel)),
		GradePay:  null.Int32From(int32(req.GradePay)),
		CreatedBy: null.StringFrom(req.CreatedBy),
		CreatedOn: null.TimeFrom(time.Now()),
		ValidFrom: null.TimeFrom(req.ValidFrom),
		ValidTo:   null.TimeFrom(req.ValidTo),
		Status:    null.StringFrom(req.Status),
		Remarks:   null.StringFrom(req.Remarks),
		GroupCode: null.Int16From(int16(req.GroupCode)),
		GroupName: null.StringFrom(req.GroupName),
	}

	createResponse, err := cmh.svc.CreateCadreMasterQuery(ctx, createCadre)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "CreateCadreMasterQuery Repo call failed: %s", err.Error())
		return
	}
	// Prepare the response
	rsp := response.NewCreateCadreMasterResponse(*createResponse)
	apiRsp := response.CreateCadreMasterAPIResponse{
		StatusCodeAndMessage: port.CreateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "CreatePostManagementMakerHandler resposne: %v", apiRsp)
	// Send the success response
	handleCreateSuccess(ctx, apiRsp)
}

type UpdateCadreRequest struct {
	CadreID   null.Int32  `uri:"cadre-id" validate:"required"`
	CadreName null.String `json:"cadre_name" validate:"required"`
	GroupName null.String `json:"group_name" validate:"required"`
	PayLevel  null.Int32  `json:"pay_level" validate:"required"`
	GradePay  null.Int32  `json:"grade_pay" validate:"required"`
	UpdatedBy null.String `json:"updated_by" validate:"required"`
	ValidFrom null.Time   `json:"valid_from" validate:"required"`
	ValidTo   null.Time   `json:"valid_to" validate:"required"`
	Status    null.String `json:"status" validate:"required"`
	Remarks   null.String `json:"remarks" validate:"required"`
	GroupCode null.Int16  `json:"group_id" validate:"required"`
}

// UpdateCadreMaster godoc
//
//	@Summary        Update Cadre Master
//	@Description    Updates a post cadre master entry.
//	@Tags           Post Management
//	@Accept         json
//	@Produce        json
//	@Param			cadre-id	path	uint32	true	"Cadre ID to update"
//	@Param          UpdateCadreRequest   body   UpdateCadreRequest  true    "cadre details to update"
//	@Success        200 {object} response.CreateCadreMasterAPIResponse "resource updated successfully"
//	@Failure        400 {object} apierrors.APIErrorResponse "Validation error"
//	@Failure        401 {object} apierrors.APIErrorResponse "Unauthorized access"
//	@Failure        403 {object} apierrors.APIErrorResponse "Forbidden access"
//	@Failure        404 {object} apierrors.APIErrorResponse "Resource not found"
//	@Failure        409 {object} apierrors.APIErrorResponse "Conflict in the data"
//	@Failure        500 {object} apierrors.APIErrorResponse "Internal server error"
//	@Router         /post-management/cadres/{cadre-id} [put]
func (cmh *CadreMasterHandler) UpdateCadreMasterHandler(ctx *gin.Context) {
	var req UpdateCadreRequest

	// Bind URI
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for UpdateCadreRequest (URI): %s", err)
		return
	}

	// Bind JSON
	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for UpdateCadreRequest (JSON): %s", err)
		return
	}

	// Validate request struct if needed
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for UpdateCadreRequest: %s", err)
		return
	}

	// Create the update object directly — no need to convert again
	updateCadre := domain.CadreMaster{
		CadreID:   req.CadreID,
		CadreName: req.CadreName,
		GroupName: req.GroupName,
		PayLevel:  req.PayLevel,
		GradePay:  req.GradePay,
		UpdatedBy: req.UpdatedBy,
		UpdatedOn: null.TimeFrom(time.Now()),
		ValidFrom: req.ValidFrom,
		ValidTo:   req.ValidTo,
		Status:    req.Status,
		Remarks:   req.Remarks,
		GroupCode: req.GroupCode,
	}

	// Call your service
	createResponse, err := cmh.svc.UpdateCadreMasterQuery(ctx, updateCadre)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "UpdateCadreMasterQuery Repo call failed: %s", err.Error())
		return
	}

	// Prepare and send the response
	rsp := response.NewCreateCadreMasterResponse(*createResponse)
	apiRsp := response.CreateCadreMasterAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "UpdateCadreMasterHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type ListAllCadreRequest struct {
	ValidFrom time.Time `form:"valid-from" time_format:"2006-01-02" validate:"required"`
	ValidTo   time.Time `form:"valid-to" time_format:"2006-01-02" validate:"required,gtfield=ValidFrom"`
	port.MetaDataRequest
}

func (cmh *CadreMasterHandler) ListAllCadresHandlerD1(ctx *gin.Context) {
	var req ListAllCadreRequest

	// Bind query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for ListAllCadreRequest: %s", err.Error())
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for ListAllCadreRequest: %s", err.Error())
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	cadreList, err := cmh.svc.ListAllCadresQryD1(ctx, req.ValidFrom, req.ValidTo, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}
	rsp := response.NewListAllCadresD1Response(cadreList)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListAllCadresD1APIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	log.Debug(ctx, "ListAllCadresHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)

}
