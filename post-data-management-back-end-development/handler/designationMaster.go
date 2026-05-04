package handler

import (
	"math"
	"pmdm/core/domain"
	"pmdm/core/port"
	"pmdm/handler/response"
	repo "pmdm/repo/postgres"
	"time"

	"github.com/gin-gonic/gin"
	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
	validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
)

// PostManagementHandler represents the HTTP handler for post management master-related requests
type DesignationMasterHandler struct {
	svc *repo.DesignationMasterRepository
}

// NewPostManagementHandler creates a new PostManagementMasterHandler instance
func NewDesignationMasterHandler(svc *repo.DesignationMasterRepository) *DesignationMasterHandler {
	return &DesignationMasterHandler{
		svc,
	}
}

// ListPostManagementMaster godoc
//
//	@Summary        ListAllDesignations
//	@Description    List all designations from the database
//	@Tags           designationMaster
//	@Accept         json
//	@Produce        json
//	@Success        200                     {object}    []DesignationMasterDetails      "List all designations from the database"
//	@Failure        400                     {object}    apierrors.APIErrorResponse          "Validation error"
//	@Failure        401                     {object}    apierrors.APIErrorResponse          "Unauthorized error"
//	@Failure        403                     {object}    apierrors.APIErrorResponse          "Forbidden error"
//	@Failure        404                     {object}    apierrors.APIErrorResponse          "Data not found error"
//	@Failure        500                     {object}    apierrors.APIErrorResponse          "Internal server error"
//	@Router         /designation/list [get]
//
// Define the handler function to list all designation master records
func (dmh *DesignationMasterHandler) ListDesignations(ctx *gin.Context) {
	// Fetch the list of all designation types using the repository method
	designationTypes, err := dmh.svc.ListDesignationsQry(ctx)
	if err != nil {
		// Handle database error
		log.Error(ctx, err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Transform the data to the desired response format
	var response []DesignationMasterDetails
	for _, designationType := range designationTypes {
		// Create a new DesignationMasterDetails instance from each domain.Designation object
		resp := NewDesignationMasterDetails(&designationType)
		response = append(response, resp)
	}

	// Handle the successful response and return the list of DesignationMasterDetails
	handleSuccess(ctx, response)
}

// ListAndFilterDesignations godoc
//
//	@Summary		Get and Filter Designations
//	@Description	Fetches the list of designations filtered by group and cadre codes. Also returns the complete list of designation types.
//	@Tags			Designation Management
//	@Accept			json
//	@Produce		json
//	@Param			group-code	query		string									false	"Group Code (optional)"
//	@Param			cadre-code	query		string									false	"Cadre Code (optional)"
//	@Success		200		{object}	response.ListAndFilterDesignationsAPIResponse	"Filtered designations and all designation types retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/designations [get]
func (de *DesignationMasterHandler) (ctx *gin.Context) {
	var req DesignationMasterListRequest

	// Bind query parameters
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}

	// Fetch the designation list by group and cadre
	designationList, err := de.svc.DesignationMasterByGroupAndCadreQuery(ctx, req.GroupCode, req.CadreCode, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Fetch the complete list of designations
	allDesignationTypes, err := de.svc.ListDesignationsQry(ctx)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Create the response data
	rsp := response.NewListAndFilterDesignationsResponse(designationList, allDesignationTypes)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.ListAndFilterDesignationsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	// Send the success response
	handleSuccess(ctx, apiRsp)
}

type GroupToCadreRequest struct {
	GroupCode       string `form:"group_code" binding:"required"`
	Limit           int    `form:"limit"`
	Skip            int    `form:"skip"`
	MetaDataRequest port.MetaDataRequest
}

type CadreToDesignationRequest struct {
	CadreCode       string `form:"cadre-code" binding:"required"`
	Limit           int    `form:"limit"`
	Skip            int    `form:"skip"`
	MetaDataRequest port.MetaDataRequest
}

// ListCadresByGroupHandler godoc
//
// @Summary      Get Cadres by Group
// @Description  Fetch a list of cadres belonging to a specific group, with optional pagination.
// @Tags         Designation Master
// @Accept       json
// @Produce      json
// @Param        group_code   query     string  true   "Group Code (e.g., A, B, C, D)"
// @Param        skip         query     int     false  "Number of records to skip (for pagination)"
// @Param        limit        query     int     false  "Number of records to return (for pagination)"
// @Success      200  {object}  response.CadreListResponse   "Cadres fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse   "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse   "Internal server error"
// @Router       /designations/cadres-by-group [get]
func (de *DesignationMasterHandler) ListCadresByGroupHandler(ctx *gin.Context) {
	var req GroupToCadreRequest

	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}

	result, err := de.svc.CadreListByGroupQuery(ctx, req.GroupCode, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	metadata := port.NewMetaDataResponse(0, 0, len(result))
	apiRsp := response.CadreListResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 result,
	}

	handleSuccess(ctx, apiRsp)
}

// ListDesignationsByCadreHandler godoc
//
// @Summary      Get Designations by Cadre
// @Description  Fetch a list of designations belonging to a specific cadre, with optional pagination.
// @Tags         Designation Master
// @Accept       json
// @Produce      json
// @Param        cadre-code   query     string  true   "Cadre Code (e.g., ENG, CLERK)"
// @Param        skip         query     int     false  "Number of records to skip (for pagination)"
// @Param        limit        query     int     false  "Number of records to return (for pagination)"
// @Success      200  {object}  response.DesignationListResponse  "Designations fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse        "Validation or binding error"
// @Failure      500  {object}  apierrors.APIErrorResponse        "Internal server error"
// @Router       /designations/designations-by-cadre [get]
func (de *DesignationMasterHandler) ListDesignationsByCadreHandler(ctx *gin.Context) {
	var req CadreToDesignationRequest

	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}

	result, err := de.svc.DesignationListByCadreQuery(ctx, req.CadreCode, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}
	rsp := response.NewListDesignationByCadreResponse(result)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.DesignationListResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 result,
	}

	handleSuccess(ctx, &apiRsp)
}

type DesignationMasterListAllRequest struct {
	port.MetaDataRequest
}

// ListAllDesignations godoc
//
//	@Summary		List All Designation Details
//	@Description	List all designation types or filter by group-code if provided
//	@Tags			Designation Management
//	@Accept			json
//	@Produce		json
//	@Param			group-code	query		string										false	"Filter designations by group code"
//	@Param       metaDataRequest	query		port.MetaDataRequest	false  		"Metadata request containing skip, limit, orderBy & sortBy"
//	@Success		200			{object}	response.ListAllDesignationsAPIResponse				"Designation list retrieved successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401			{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403			{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404			{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409			{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500			{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/designations/all-designations [get]
func (cmh *DesignationMasterHandler) ListAllDesignationsHandler(ctx *gin.Context) {
	var req DesignationMasterListAllRequest

	// Bind query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for DesignationMasterListAllRequest: %s", err.Error())
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for DesignationMasterListAllRequest: %s", err.Error())
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	designationList, err := cmh.svc.ListAllDesignationsQry(ctx, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}
	rsp := response.NewListAllDesignationsResponse(designationList)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListAllDesignationsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	log.Debug(ctx, "ListAllDesignationsHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)

}

type CreateDesignationRequest struct {
	Designation string    `json:"designation" db:"designation" validate:"required"`
	GroupName   string    `json:"group_name" db:"group_name" validate:"required"`
	CadreName   string    `json:"cadre_name" db:"cadre_name" validate:"required"`
	CreatedBy   string    `json:"created_by" db:"created_by" validate:"required"`
	ValidFrom   time.Time `json:"valid_from" db:"valid_from" validate:"required"`
	ValidTo     time.Time `json:"valid_to" db:"valid_to" validate:"required"`
	Status      string    `json:"status" db:"status" validate:"required"`
	Remarks     string    `json:"remarks" db:"remarks" validate:"required"`
	CadreId     int       `json:"cadre_id" db:"cadre_id" validate:"required"`
	GroupId     int16     `json:"group_id" db:"group_id" validate:"required"`
}

// CreateDesignationMaster godoc
//
//	@Summary        Create Designation Master
//	@Description    Creates a post designation master entry.
//	@Tags           Designation Management
//	@Accept         json
//	@Produce        json
//	@Param          CreateDesignationRequest   body   CreateDesignationRequest  true    "designation details to create"
//	@Success        200 {object} response.CreateDesignationMasterAPIResponse "Designation master details created successfully"
//	@Failure        400 {object} apierrors.APIErrorResponse "Validation error"
//	@Failure        401 {object} apierrors.APIErrorResponse "Unauthorized access"
//	@Failure        403 {object} apierrors.APIErrorResponse "Forbidden access"
//	@Failure        404 {object} apierrors.APIErrorResponse "Resource not found"
//	@Failure        409 {object} apierrors.APIErrorResponse "Conflict in the data"
//	@Failure        500 {object} apierrors.APIErrorResponse "Internal server error"
//	@Router         /post-management/designations/create [post]
func (cmh *DesignationMasterHandler) CreateDesignationMasterHandler(ctx *gin.Context) {
	var req CreateDesignationRequest
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

	createDesignation := domain.DesignationMaster{
		Designation: req.Designation,
		GroupName:   req.GroupName,
		CadreName:   req.CadreName,
		CreatedBy:   req.CreatedBy,
		CreatedDate: time.Now(),
		ValidFrom:   req.ValidFrom,
		ValidTo:     req.ValidTo,
		Status:      req.Status,
		Remarks:     req.Remarks,
		CadreId:     req.CadreId,
		GroupId:     req.GroupId,
	}

	createResponse, err := cmh.svc.CreateDesignationMasterQuery(ctx, createDesignation)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "CreateDesignationMasterQuery Repo call failed: %s", err.Error())
		return
	}
	// Prepare the response
	rsp := response.NewCreateDesignationMasterResponse(*createResponse)
	apiRsp := response.CreateDesignationMasterAPIResponse{
		StatusCodeAndMessage: port.CreateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "CreatePostManagementMakerHandler resposne: %v", apiRsp)
	// Send the success response
	handleCreateSuccess(ctx, apiRsp)
}

type UpdateDesignationRequest struct {
	DesignationID  int       `json:"designation_id" db:"designation_id" validate:"required"`
	Designation    string    `json:"designation" db:"designation" validate:"required"`
	GroupName      string    `json:"group_name" db:"group_name" validate:"required"`
	CadreName      string    `json:"cadre_name" db:"cadre_name" validate:"required"`
	ValidFrom      time.Time `json:"valid_from" db:"valid_from" validate:"required"`
	ValidTo        time.Time `json:"valid_to" db:"valid_to" validate:"required"`
	Status         string    `json:"status" db:"status" validate:"required"`
	Remarks        string    `json:"remarks" db:"remarks" validate:"required"`
	CadreId        int       `json:"cadre_id" db:"cadre_id" validate:"required"`
	GroupId        int16     `json:"group_id" db:"group_id" validate:"required"`
	DesignationUID int       `uri:"designation-uid" db:"designation_uid" validate:"required"`
}

// UpdateDesignationMaster godoc
//
//	@Summary        Update Designation Master
//	@Description    Updates a post designation master entry.
//	@Tags           Designation Management
//	@Accept         json
//	@Produce        json
//	@Param			designation-uid	path	uint32	true	"Designation UID to update"
//	@Param          UpdateDesignationRequest   body   UpdateDesignationRequest  true    "designation details to update"
//	@Success        200 {object} response.CreateDesignationMasterAPIResponse "resource updated successfully"
//	@Failure        400 {object} apierrors.APIErrorResponse "Validation error"
//	@Failure        401 {object} apierrors.APIErrorResponse "Unauthorized access"
//	@Failure        403 {object} apierrors.APIErrorResponse "Forbidden access"
//	@Failure        404 {object} apierrors.APIErrorResponse "Resource not found"
//	@Failure        409 {object} apierrors.APIErrorResponse "Conflict in the data"
//	@Failure        500 {object} apierrors.APIErrorResponse "Internal server error"
//	@Router         /post-management/designations/{designation-uid} [put]
func (cmh *DesignationMasterHandler) UpdateDesignationMasterHandler(ctx *gin.Context) {
	var req UpdateDesignationRequest

	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for UpdateDesignationRequest (URI): %s", err)
		return
	}

	if err := ctx.ShouldBindJSON(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for UpdateDesignationRequest (JSON): %s", err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for UpdateDesignationRequest: %s", err)
		return
	}

	updateDesignation := domain.DesignationMaster{
		DesignationID:  req.DesignationID,
		Designation:    req.Designation,
		GroupName:      req.GroupName,
		CadreName:      req.CadreName,
		ValidFrom:      req.ValidFrom,
		ValidTo:        req.ValidTo,
		Status:         req.Status,
		Remarks:        req.Remarks,
		CadreId:        req.CadreId,
		GroupId:        req.GroupId,
		DesignationUID: req.DesignationUID,
	}

	// Call your service
	createResponse, err := cmh.svc.UpdateDesignationMasterQuery(ctx, updateDesignation)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "UpdateDesignationMasterQuery Repo call failed: %s", err.Error())
		return
	}

	// Prepare and send the response
	rsp := response.NewCreateDesignationMasterResponse(*createResponse)
	apiRsp := response.CreateDesignationMasterAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "UpdateDesignationMasterHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type DestignationMasterListAllRequest struct {
	ValidFrom time.Time `form:"valid-from" time_format:"2006-01-02" validate:"required"`
	ValidTo   time.Time `form:"valid-to" time_format:"2006-01-02" validate:"required,gtfield=ValidFrom"`
	port.MetaDataRequest
}

func (cmh *DesignationMasterHandler) ListDesignationD1(ctx *gin.Context) {
	var req DestignationMasterListAllRequest

	// Bind query parameters to the struct
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for DesignationMasterListAllRequest: %s", err.Error())
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for DesignationMasterListAllRequest: %s", err.Error())
		return
	}

	if req.Limit == 0 && req.Skip == 0 {
		req.Limit = math.MaxInt32
	}
	designationList, err := cmh.svc.ListAllDesignationsD1(ctx, req.ValidFrom, req.ValidTo, req.MetaDataRequest)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}
	rsp := response.NewListAllDesignationsResponse(designationList)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ListAllDesignationsAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	log.Debug(ctx, "ListAllDesignationsHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)

}
