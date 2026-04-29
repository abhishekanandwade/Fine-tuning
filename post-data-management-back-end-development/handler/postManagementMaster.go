package handler

import (
	"fmt"
	"net/http"
	"pmdm/core/domain"
	"pmdm/core/port"
	"pmdm/handler/response"
	repo "pmdm/repo/postgres"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/minio/minio-go/v7"
	config "gitlab.cept.gov.in/it-2.0-common/api-config"
	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
	apiutility "gitlab.cept.gov.in/it-2.0-common/api-utility"
	validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
)

// PostManagementHandler represents the HTTP handler for post management master-related requests
type PostManagementHandler struct {
	svc         *repo.PostManagementRepository
	Config      *config.Config
	MinioClient *minio.Client
}

// NewPostManagementHandler creates a new PostManagementMasterHandler instance
func NewPostManagementHandler(svc *repo.PostManagementRepository, Config *config.Config, MinioClient *minio.Client) *PostManagementHandler {
	return &PostManagementHandler{
		svc,
		Config,
		MinioClient,
	}
}

// PostManagementCreateRequest represents a request body for creating post management master details
type PostManagementCreateRequest struct {
	OfficeID                  int       `json:"office_id" validate:"required,office_id"`
	PostName                  string    `json:"post_name" validate:"required"`
	OfficeName                string    `json:"office_name" validate:"required"`
	GroupId                   int       `json:"group_id" validate:"required"`
	CadreID                   int       `json:"cadre_id" validate:"required"`
	CadreName                 string    `json:"cadre_name" validate:"required"`
	AllowancesAttached        bool      `json:"allowances_attached"`
	AllowanceDescription      string    `json:"allowance_description"`
	CreatedBy                 string    `json:"created_by" validate:"required"`
	OrderCaseMark             string    `json:"order_casemark" validate:"required"`
	OrderDate                 time.Time `json:"order_date" validate:"required"`
	UploadOrderDocName        string    `json:"upload_order_doc_name" validate:"required"`
	EstablishmentRegisterID   int       `json:"establishment_register_id" validate:"required"`
	Designation               string    `json:"designation" validate:"required"`
	PayLevel                  int       `json:"pay_level" validate:"required"`
	GradePay                  int       `json:"grade_pay" validate:"required"`
	PermanentStatus           bool      `json:"permanent_status" validate:"required"`
	EstablishmentRegisterName string    `json:"establishment_register_name" validate:"required"`
	EmployeeGroup             string    `json:"employee_group" validate:"required"`
	SanctionedStrength        int       `json:"sanctioned_strength" validate:"required"`
	ApprovePostID             string    `json:"approve_post_id"`
	OfficeType                string    `json:"office_type" validate:"required"`
	GroupName                 string    `json:"group_name" validate:"required"`
	DesignationId             int       `json:"designation_id" validate:"required"`
}

type PostManagementCreateRequests struct {
	PostCreateReq []PostManagementCreateRequest `json:"postcreatereq"`
}

// Example function to generate a unique ID
func generateUniqueID() string {
	return strconv.FormatInt(time.Now().UnixNano(), 10) // Example using current time in nanoseconds
}

// PostManagementListRequest represents a request body for listing post management master details
type PostManagementListRequest11 struct {
	OfficeID int   `uri:"office-id" validate:"required"`
	RAPostID int64 `uri:"post-id" validate:"required"`
}
type PostManagementListWithOfficeIDRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

type PostManagementListWithOfficeIDForMakerRequest struct {
	OfficeID int    `form:"office-id" validate:"required"`
	PostID   string `form:"post-id,omitempty"`
}

type PostManagementListRequest123 struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

// PostManagementFetchRequest represents a request body for listing post management details
type PostManagementFetchRequest struct {
	OfficeID     int    `uri:"office-id,omitempty"`
	FilledStatus string `form:"filled-status,omitempty"`
	GroupID      int    `form:"group-id,omitempty"`
}

// PostManagementByOfficeIDAndStatus godoc
//
//	@Summary		Get Post Management by Office ID and Filled Status
//	@Description	Fetches post management details by Office ID, filled status, and group ID
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			office-id		path		int										true	"Office ID"
//	@Param			filled-status	query		string									true	"Filled Status (filled/unfilled)"
//	@Param			group-id		query		int										true	"Group ID"
//	@Success		200		{object}	response.PostManagementByOfficeIDAndStatusAPIResponse	"Post management details retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse								"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse								"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse								"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse								"Data not found error"
//	@Failure		409		{object}	apierrors.APIErrorResponse								"Data conflict error"
//	@Failure		500		{object}	apierrors.APIErrorResponse								"Internal server error"
//	@Router			/post-management/office-post-details/{office-id}/posts [get]
func (pmh *PostManagementHandler) PostManagementByOfficeIDAndStatusHandler(ctx *gin.Context) {
	var req PostManagementFetchRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementFetchRequest: %s", err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementFetchRequest: %s", err)
		return
	}

	postManagementList, err := pmh.svc.PostManagementByOfficeIDAndStatusQuery(ctx, req.OfficeID, req.FilledStatus, req.GroupID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementByOfficeIDAndStatusQuery Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewPostManagementByOfficeIDAndStatusResponse(postManagementList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.PostManagementByOfficeIDAndStatusAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementByOfficeIDAndStatusHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type PostManagementByCadreAndOfficeRequest struct {
	OfficeID  int    `uri:"office-id" validate:"required"`
	CadreName string `uri:"cadre-name" validate:"required"`
	port.MetaDataRequest
}

type PostManagementGroupByOfficeRequest struct {
	OfficeID int `form:"office-id" validate:"required"`
}

// PostManagementGroupByCadreCountByOfficeID godoc
//
//	@Summary		Get Post Management Cadre Count Grouped by Office ID
//	@Description	Fetches the count of posts grouped by cadre for a specific Office ID
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			office-id	query		string										true	"Office ID"
//	@Success		200			{object}	response.PostManagementGroupByCadreCountByOfficeIDAPIResponse	"Post count by cadre retrieved successfully"
//	@Failure		400			{object}	apierrors.APIErrorResponse							"Validation error"
//	@Failure		401			{object}	apierrors.APIErrorResponse							"Unauthorized error"
//	@Failure		403			{object}	apierrors.APIErrorResponse							"Forbidden error"
//	@Failure		404			{object}	apierrors.APIErrorResponse							"Data not found error"
//	@Failure		409			{object}	apierrors.APIErrorResponse							"Data conflict error"
//	@Failure		500			{object}	apierrors.APIErrorResponse							"Internal server error"
//	@Router			/post-management/cadres/cadre-groups [get]
func (pmh *PostManagementHandler) PostManagementGroupByCadreCountByOfficeIDHandler(ctx *gin.Context) {
	var req PostManagementGroupByOfficeRequest
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for PostManagementGroupByOfficeRequest: %s", err)
		return
	}

	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for PostManagementGroupByOfficeRequest: %s", err)
		return
	}
	postList, err := pmh.svc.PostManagementGroupByCadreCountByOfficeID(ctx, req.OfficeID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "PostManagementGroupByCadreCountByOfficeID Repo call failed: %s", err.Error())
		return
	}

	rsp := response.NewPostManagementGroupByCadreCountByOfficeIDResponse(postList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.PostManagementGroupByCadreCountByOfficeIDAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "PostManagementGroupByCadreCountByOfficeIDHandler resposne: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// GetPostDetailsbyPostIDHandler retrieves the employee details based on post id.
// @Summary      fetches the details of employee details basesd on post id.
// @Description  fetches the details of employee details basesd on post id.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param post-id path string true "Post ID"
// @Success 200 {object} response.GetPostDetailsbyPostIDAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /posts/post-details/:post-id [get]
func (pmh *PostManagementHandler) GetPostDetailsbyPostIDHandler(ctx *gin.Context) {

	var req GetPostDetailsbyPostIDRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostDetailsbyPostID(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Cadre details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetPostDetailsbyPostIDResponse(postsavbl)

	apiRsp := response.GetPostDetailsbyPostIDAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostDetailsbyPostID API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostDetailsbyPostIDRequest struct {
	PostID int `uri:"post-id" validate:"required"`
}

// GetPostsFilledVacantStatusHandler retrieves the posts status based on office id.
// @Summary      fetches the count of filled and vacant posts basesd on office id.
// @Description  fetches the count of filled and vacant posts basesd on office id.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param office-id path string true "Office ID"
// @Success 200 {object} response.GetPostsFilledVacantStatusAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /reports/posts-filled-vacant-status/:office-id [get]
func (pmh *PostManagementHandler) GetPostsFilledVacantStatusHandler(ctx *gin.Context) {

	var req GetPostsFilledVacantStatusRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostsFilledVacantStatus(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Cadre details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetPostsFilledVacantStatusResponse(postsavbl)

	apiRsp := response.GetPostsFilledVacantStatusAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostsFilledVacantStatus API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostsFilledVacantStatusRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

// GetPostsCreatedRedeployedAbolishedHandler retrieves the count of posts created, redeployed and abolished.
// @Summary      fetches the count of posts created, redeployed and abolished.
// @Description  fetches the count of posts created, redeployed and abolished.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param year path string true "year"
// @Param month path string true "month"
// @Success 200 {object} response.GetPostsCreatedRedeployedAbolishedAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /reports/posts-created-redeployed-abolished/:year/:month [get]
func (pmh *PostManagementHandler) GetPostsCreatedRedeployedAbolishedHandler(ctx *gin.Context) {
	var req GetPostsCreatedRedeployedAbolishedRequest
	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	totals, err := pmh.svc.GetPostsCreatedRedeployedAbolished(ctx, req.Year, req.Month)
	if err != nil {
		log.Error(ctx, "Database error while fetching posts created/redeployed/abolished:", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	rsp := response.NewGetPostsCreatedRedeployedAbolishedResponse(totals)
	apiRsp := response.GetPostsCreatedRedeployedAbolishedAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}
	log.Debug(ctx, "GetPostsCreatedRedeployedAbolished API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostsCreatedRedeployedAbolishedRequest struct {
	Year  int `uri:"year" validate:"required"`
	Month int `uri:"month" validate:"required"`
}

// GetPostsFilledVacantStatusDetailedHandler retrieves the posts filled and vacant status based on office id.
// @Summary      fetches the details of posts filled and vacant status based on office id.
// @Description  fetches the details of posts filled and vacant status based on office id.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param office-id path string true "Office ID"
// @Success 200 {object} response.GetPostsFilledVacantStatusAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /reports/posts-filled-vacant-status-detailed/:office-id [get]
func (pmh *PostManagementHandler) GetPostsFilledVacantStatusDetailedHandler(ctx *gin.Context) {

	var req GetPostsFilledVacantStatusDetailedRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostsFilledVacantStatusDetailed(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Cadre details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetPostsFilledVacantStatusDetailedResponse(postsavbl)

	apiRsp := response.GetPostsFilledVacantStatusDetailedAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostsFilledVacantStatus API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostsFilledVacantStatusDetailedRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

// UpdatePostDetailsbyPostIDHandler for updating the post details by post id.
// @Summary      for updating the post details by post id.
// @Description  for updating the post details by post id.
// @Tags         Posts
// @Accept       json
// @Produce      json
// @Param request body UpdatePostDetailsbyPostIDRequest true "Post details to be updated"
// @Success 201 {object} response.UpdatePostDetailsbyPostIDAPIResponse "Success - Post redeployed successfully"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /posts/post-details-update [put]
func (pmh *PostManagementHandler) UpdatePostDetailsbyPostIDHandler(ctx *gin.Context) {

	var req UpdatePostDetailsbyPostIDRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, false, true); err != nil {
		log.Error(ctx, err)
		return
	}
	//check if post id avaibale in post management master table
	postsavbl, err := pmh.svc.GetPostDetailsbyPostID(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "error while fetching post details", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}
	if postsavbl == nil {
		log.Error(ctx, "post id not found", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}
	postManagementMasterUpdate := ToPostManagementMasterUpdate2(req)

	_, err = pmh.svc.UpdatePostDetailsbyPostIDRepo(ctx, postManagementMasterUpdate)
	if err != nil {
		log.Error(ctx, "Database error while post updation", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.UpdatePostDetailsbyPostIDAPIResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 "Post details updated successfully",
	}

	log.Debug(ctx, "UpdatePostDetailsbyPostID API response:  %s", apiRsp)
	handleCreateSuccess(ctx, apiRsp)
}

type ExceptionReportOrderCasemarkRequest struct {
	port.MetaDataRequest
}

type ExceptionReportOrderCasemarkConsolidatedRequest struct {
	CircleOfficeID   *int `form:"circle_office_id"`
	RegionOfficeID   *int `form:"region_office_id"`
	DivisionOfficeID *int `form:"division_office_id"`
	Skip             int  `form:"skip"`
	Limit            int  `form:"limit"`
}

// ExceptionReportOrderCasemarkHandler godoc
//
// @Summary      Get Exception Report for Order Casemark
// @Description  Retrieves a paginated exception report for order casemark, useful for identifying inconsistencies or errors.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        skip   query     int     false  "Number of records to skip for pagination"
// @Param        limit  query     int     false  "Maximum number of records to return"
// @Success      200  {object}  response.ExceptionReportOrderCasemarkAPIResponse "Report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/exception-report-order-casemark [get]
func (pmh *PostManagementHandler) ExceptionReportOrderCasemarkHandler(ctx *gin.Context) {
	var req ExceptionReportOrderCasemarkRequest
	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	report, err := pmh.svc.ExceptionReportOrderCasemark(ctx, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "Database error while fetching Exception Report Order Casemark:", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewExceptionReportOrderCasemarkResponse(report)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
	apiRsp := response.ExceptionReportOrderCasemarkAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}
	log.Debug(ctx, "ExceptionReportOrderCasemarkHandler API Response: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// func (pmh *PostManagementHandler) ExceptionReportEstablishmentRegisterHandler(ctx *gin.Context) {
// 	var req ExceptionReportOrderCasemarkRequest
// 	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
// 		log.Error(ctx, err)
// 		return
// 	}

// 	report, err := pmh.svc.EstablishmentRegister(ctx, req.MetaDataRequest)
// 	if err != nil {
// 		log.Error(ctx, "Database error while fetching Exception Report EstablishmentRegister:", err.Error())
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}

// 	rsp := response.NewExceptionReportOrderCasemarkResponse(report)
// 	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
// 	apiRsp := response.ExceptionReportOrderCasemarkAPIResponse{
// 		StatusCodeAndMessage: port.ListSuccess,
// 		MetaDataResponse:     metadata,
// 		Data:                 rsp,
// 	}
// 	log.Debug(ctx, "ExceptionReportEstablishmentRegisterHandler API Response to be sent: ", apiRsp)
// 	handleSuccess(ctx, apiRsp)
// }

// func (pmh *PostManagementHandler) ExceptionReportOfficeNameHandler(ctx *gin.Context) {
// 	var req ExceptionReportOrderCasemarkRequest
// 	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
// 		log.Error(ctx, err)
// 		return
// 	}

// 	report, err := pmh.svc.ExceptionReportOfficeName(ctx, req.OfficeID, req.MetaDataRequest)
// 	if err != nil {
// 		log.Error(ctx, "Database error while fetching Exception Report Office Name:", err.Error())
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}

// 	rsp := response.NewExceptionReportOrderCasemarkResponse(report)
// 	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
// 	apiRsp := response.ExceptionReportOrderCasemarkAPIResponse{
// 		StatusCodeAndMessage: port.ListSuccess,
// 		MetaDataResponse:     metadata,
// 		Data:                 rsp,
// 	}
// 	log.Debug(ctx, "ExceptionReportOfficeNameHandler API Response to be sent: ", apiRsp)
// 	handleSuccess(ctx, apiRsp)
// }

// func (pmh *PostManagementHandler) ExceptionReportCadreNameHandler(ctx *gin.Context) {
// 	var req ExceptionReportOrderCasemarkRequest
// 	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
// 		log.Error(ctx, err)
// 		return
// 	}

// 	report, err := pmh.svc.ExceptionReportCadreName(ctx, req.OfficeID, req.MetaDataRequest)
// 	if err != nil {
// 		log.Error(ctx, "Database error while fetching Exception Report Cadre Name:", err.Error())
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}

// 	rsp := response.NewExceptionReportOrderCasemarkResponse(report)
// 	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
// 	apiRsp := response.ExceptionReportOrderCasemarkAPIResponse{
// 		StatusCodeAndMessage: port.ListSuccess,
// 		MetaDataResponse:     metadata,
// 		Data:                 rsp,
// 	}
// 	log.Debug(ctx, "ExceptionReportCadreNameHandler API Response to be sent: ", apiRsp)
// 	handleSuccess(ctx, apiRsp)
// }

// func (pmh *PostManagementHandler) ExceptionReportGroupNameHandler(ctx *gin.Context) {
// 	var req ExceptionReportOrderCasemarkRequest
// 	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
// 		log.Error(ctx, err)
// 		return
// 	}

// 	report, err := pmh.svc.ExceptionReportGroupName(ctx, req.OfficeID, req.MetaDataRequest)
// 	if err != nil {
// 		log.Error(ctx, "Database error while fetching Exception Report Group Name:", err.Error())
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}

// 	rsp := response.NewExceptionReportOrderCasemarkResponse(report)
// 	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsp))
// 	apiRsp := response.ExceptionReportOrderCasemarkAPIResponse{
// 		StatusCodeAndMessage: port.ListSuccess,
// 		MetaDataResponse:     metadata,
// 		Data:                 rsp,
// 	}
// 	log.Debug(ctx, "ExceptionReportGroupNameHandler API Response to be sent: ", apiRsp)
// 	handleSuccess(ctx, apiRsp)
// }

// GetCadreWiseReportsHandler retrieves cadre wise reports based on office id.
// @Summary      fetches the cadre wise reports based on office id.
// @Description  fetches the cadre wise reports based on office id.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param office-id path string true "Office ID"
// @Success 200 {object} response.GetCadreWiseReportsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /reports/cadre-wise-reports/:office-id [get]
func (pmh *PostManagementHandler) GetCadreWiseReportsHandler(ctx *gin.Context) {

	var req GetCadreWiseReportsRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetCadreWiseReports2(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Cadre wise reports:", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetCadreWiseReportsResponse(postsavbl)

	apiRsp := response.GetCadreWiseReportsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostsFilledVacantStatus API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetCadreWiseReportsRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

// GetCadreWiseReportsHandler retrieves cadre wise reports based on office id and cadre id.
// @Summary      fetches the cadre wise reports based on office id and cadre id.
// @Description  fetches the cadre wise reports based on office id and cadre id.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param office-id path string true "Office ID"
// @Param cadre-id path string true "Cadre ID"
// @Success 200 {object} response.GetCadreWiseofficewiseReportsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /reports/cadre-wise-reports/:office-id/:cadre-id [get]
func (pmh *PostManagementHandler) GetCadreWiseOfficeWiseReportsHandler(ctx *gin.Context) {

	var req GetCadreWiseOfficeWiseReportsRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetCadreWiseOfficeWiseReports(ctx, req.OfficeID, req.CadreID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Cadre wise reports:", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetCadreWiseOfficeWiseReportsResponse(postsavbl)

	apiRsp := response.GetCadreWiseofficewiseReportsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostsFilledVacantStatus API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetCadreWiseOfficeWiseReportsRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
	CadreID  int `uri:"cadre-id" validate:"required"`
}

type CadreReportRequest struct {
	CadreId     int64  `form:"cadre-id" validation:"required"`
	Search      string `form:"search"`
	IncludeList bool   `form:"includelist"`
	port.MetaDataRequest
}

// GetCadreReport godoc
//
// @Summary      Get Cadre Report
// @Description  Fetches cadre summary, counts, and optional detail list for a given cadre.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        cadre-id     query     int     true   "Cadre ID (required)"
// @Param        search       query     string  false  "Search filter applied on cadre data"
// @Param        includelist  query     bool    false  "Whether to include detailed list (true/false)"
// @Param        skip         query     int     false  "Number of records to skip for pagination"
// @Param        limit        query     int     false  "Maximum number of records to return"
// @Success      200  {object}  response.CadreReportResponse "Cadre report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/cadres [get]
func (pmh *PostManagementHandler) GetCadreReport(ctx *gin.Context) {
	var req CadreReportRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	summaryData, err := pmh.svc.GetCadreSummary(ctx, req.CadreId, req.Search, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "DB error in GetCadreSummary:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	totalCount, err := pmh.svc.GetTotalCircleCount(ctx, req.CadreId, req.Search)
	if err != nil {
		log.Error(ctx, "DB error in GetTotalCircleCount:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	var totalPosts, totalFilled, TotalVacant int
	for _, row := range summaryData {
		totalPosts += row.TotalPosts
		totalFilled += row.TotalFilled
		TotalVacant += row.TotalVacant
	}

	var detailList []domain.Detail
	if req.IncludeList {
		detailList, err = pmh.svc.GetDetailList(ctx, req.CadreId, req.Search)
		if err != nil {
			log.Error(ctx, "DB error in GetDetailList", err)
			apierrors.HandleDBError(ctx, err)
			return
		}
	}

	var cadreInfo domain.CadreInfo
	if len(summaryData) > 0 {
		cadreInfo = domain.CadreInfo{
			CadreID:   req.CadreId,
			CadreName: summaryData[0].CadreName,
			GroupName: summaryData[0].GroupName,
		}
	}

	summaryResponse := response.NewSummaryResponses(summaryData)
	detailResponse := response.NewDetailResponses(detailList)
	cadreInfoResponse := response.NewCadreInfoResponse(cadreInfo)
	countSummaryResponse := response.NewCountSummaryResponse(totalPosts, totalFilled, TotalVacant)

	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, totalCount)

	rsp := response.CadreReportResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 summaryResponse,
		Summary:              countSummaryResponse,
		CadreInfo:            cadreInfoResponse,
		DetailList:           detailResponse,
	}
	handleSuccess(ctx, rsp)
}

type CircleCadreReportRequest struct {
	CadreName      string `form:"cadre-name" validation:"required"`
	CircleOfficeID int64  `form:"circle-office-id"`
	Search         string `form:"search"`
	IncludeList    bool   `form:"includelist"`
	port.MetaDataRequest
}

// GetCircleCadreReport godoc
//
// @Summary      Get Circle Cadre Report
// @Description  Fetches summary, counts, and optional detail list for cadres within a circle.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        cadre-name        query     string  true   "Cadre Name (required)"
// @Param        circle-office-id  query     int     false  "Circle Office ID (filter within specific circle)"
// @Param        search            query     string  false  "Search filter applied on cadre data"
// @Param        includelist       query     bool    false  "Whether to include detailed post list (true/false)"
// @Param        skip              query     int     false  "Number of records to skip for pagination"
// @Param        limit             query     int     false  "Maximum number of records to return"
// @Success      200  {object}  response.CircleCadreReportResponse "Circle Cadre report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/circle [get]
func (pmh *PostManagementHandler) GetCircleCadreReport(ctx *gin.Context) {
	var req CircleCadreReportRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	summaryData, err := pmh.svc.GetCircleSummary(ctx, req.CadreName, req.Search, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "DB error in GetCircleSummary:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	totalCount, err := pmh.svc.GetTotalCircleCadreCount(ctx, req.CadreName, req.Search)
	if err != nil {
		log.Error(ctx, "DB error in GetTotalCircleCadreCount:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	var totalPost, totalFilled, totalVacant int
	for _, row := range summaryData {
		totalPost += row.TotalPosts
		totalFilled += row.TotalFilledPosts
		totalVacant += row.TotalVacantPosts
	}

	var detailList []domain.DetailedPost
	if req.IncludeList {
		detailList, err = pmh.svc.GetCircleDetailList(ctx, req.CadreName, req.CircleOfficeID, req.Search)
		if err != nil {
			log.Error(ctx, "DB error in GetCircleDetailList:", err)
			apierrors.HandleDBError(ctx, err)
			return
		}
	}

	var hierarchyinfo domain.HierarchyInfo
	if len(summaryData) > 0 {
		hierarchyinfo = domain.HierarchyInfo{
			Level:     1,
			LevelName: "Circle",
			CadreName: summaryData[0].CadreName,
		}
	}

	summaryResponse := response.NewCircleSummaryResponse(summaryData)
	detailsResponse := response.NewDetailPostResponse(detailList)
	hierarchyinfoResponse := response.NewHierarchyInfoResponse(hierarchyinfo)
	countResponse := response.NewCountSummaryResponse(totalPost, totalFilled, totalVacant)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, totalCount)

	rsp := response.CircleCadreReportResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 summaryResponse,
		Summary:              countResponse,
		HierarchyInfo:        hierarchyinfoResponse,
		DetailList:           detailsResponse,
	}
	handleSuccess(ctx, rsp)
}

type DivisionReportRequest struct {
	CadreName        string `form:"cadre_name" validation:"required"`
	RegionOfficeId   int64  `form:"region_office_id" validation:"required"`
	DivisionOfficeId int64  `form:"division_office_id"`
	Search           string `form:"search"`
	IncludeList      bool   `form:"includelist"`
	port.MetaDataRequest
}

// GetDivisionsReport godoc
//
// @Summary      Get Division Cadre Report
// @Description  Fetches cadre summaries, counts, region info, and optionally detail list for divisions under a region.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        cadre_name         query     string  true   "Cadre Name (required)"
// @Param        region_office_id   query     int     true   "Region Office ID (required)"
// @Param        division_office_id query     int     false  "Division Office ID (filter specific division)"
// @Param        search             query     string  false  "Search filter applied on division data"
// @Param        includelist        query     bool    false  "Whether to include detailed division list (true/false)"
// @Param        skip               query     int     false  "Number of records to skip for pagination"
// @Param        limit              query     int     false  "Maximum number of records to return"
// @Success      200  {object}  response.DivisionCadreReportResponse "Division Cadre report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/divisions [get]
func (pmh *PostManagementHandler) GetDivisionsReport(ctx *gin.Context) {
	var req DivisionReportRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	summaries, count, err := pmh.svc.GetDivisionSummaries(ctx, req.CadreName, req.RegionOfficeId, req.Search, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "DB error in GetDivisionSummaries:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	summary := map[string]int{
		"totalPosts":  0,
		"totalFilled": 0,
		"totalVacant": 0,
	}
	for _, s := range summaries {
		summary["totalPosts"] += s.TotalPosts
		summary["totalFilled"] += s.TotalFilledPosts
		summary["totalVacant"] += s.TotalVacantPosts
	}

	regionInfo, err := pmh.svc.GetRegionInfo(ctx, req.RegionOfficeId)
	if err != nil {
		log.Error(ctx, "DB error in GetRegionInfo:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	infores := domain.RegionInfo{
		Level:          3,
		LevelName:      "Divisions",
		CadreName:      req.CadreName,
		RegionOfficeId: req.RegionOfficeId,
		RegionName:     regionInfo.RegionName,
		CircleOfficeID: regionInfo.CircleOfficeID,
		CircleName:     regionInfo.CircleName,
	}

	var detailList []domain.DivisionDetail
	if req.IncludeList && req.DivisionOfficeId != 0 {
		detailList, err = pmh.svc.GetDivisionDetail(ctx, req.CadreName, req.RegionOfficeId, req.DivisionOfficeId, req.Search)
		if err != nil {
			log.Error(ctx, "DB error in GetDivisionDetail:", err)
			apierrors.HandleDBError(ctx, err)
			return
		}
	}

	divisionSummary := response.NewDivisionSummaryResponse(summaries)
	countSummary := response.NewCountSummaryResponse(summary["totalPosts"], summary["totalFilled"], summary["totalVacant"])
	regionInfoRsp := response.NewRegionInfoResponse(infores)
	detailListRsp := response.NewDetailListResponse(detailList)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, count)

	rsp := response.DivisionCadreReportResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		MetaDataResponse:     metadata,
		Data:                 divisionSummary,
		Summary:              countSummary,
		HierarchyInfo:        regionInfoRsp,
		DetailList:           detailListRsp,
	}

	handleSuccess(ctx, rsp)
}

type HierarchyReportParams struct {
	CadreName      string `form:"cadre-name"`
	ParentOfficeID int64  `form:"parent-office-id"`
	Search         string `form:"search"`
	IncludeList    bool   `form:"includelist"`
	port.MetaDataRequest
}

// GetHierarchyReport godoc
//
// @Summary      Get Hierarchy Cadre Report
// @Description  Retrieves hierarchy-wise cadre summaries, counts, hierarchy info, and optionally detailed posts.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        cadre-name        query     string  false  "Cadre name (optional filter)"
// @Param        parent-office-id  query     int     true   "Parent Office ID (required for hierarchy level)"
// @Param        search            query     string  false  "Search filter"
// @Param        includelist       query     bool    false  "Include detailed hierarchy list (true/false)"
// @Param        skip              query     int     false  "Number of records to skip for pagination"
// @Param        limit             query     int     false  "Maximum number of records to return"
// @Success      200  {object}  response.HierarchyReportResponse "Hierarchy Cadre report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/hierarchy [get]
func (pmh *PostManagementHandler) GetHierarchyReport(ctx *gin.Context) {
	var req HierarchyReportParams

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	summaries, total, err := pmh.svc.GetHierarchySummaries(ctx, req.CadreName, req.ParentOfficeID, req.Search, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "DB error in GetHierarchyReport:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	summary := map[string]int{
		"totalPosts":  0,
		"totalFilled": 0,
		"totalVacant": 0,
	}
	for _, s := range summaries {
		summary["totalPosts"] += s.TotalPosts
		summary["totalFilled"] += s.TotalFilledPosts
		summary["totalVacant"] += s.TotalVacantPosts
	}

	var details []domain.HierarchyDetail
	if req.IncludeList {
		details, err = pmh.svc.GetHierarchyDetailList(ctx, req.CadreName, req.ParentOfficeID, req.Search)
		if err != nil {
			log.Error(ctx, "DB error in GetHierarchyReport:", err)
			apierrors.HandleDBError(ctx, err)
			return
		}
	}

	hierachyInfo, err := pmh.svc.GetHierarchyInfo(ctx, req.ParentOfficeID)
	if err != nil {
		log.Error(ctx, "DB error in GetHierarchyInfo:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}
	infores := domain.HierarchyInfodata{
		ParentOfficeID:   hierachyInfo.ParentOfficeID,
		ParentOfficeName: hierachyInfo.ParentOfficeName,
		CadreName:        req.CadreName,
		Level: func() int {
			if req.ParentOfficeID == 35320001 {
				return 1
			}
			return 2
		}(),
	}

	hierachySummary := response.NewHierarchySummaryResponse(summaries)
	countSummary := response.NewCountSummaryResponse(summary["totalPosts"], summary["totalFilled"], summary["totalVacant"])
	hierachyInfoRsp := response.NewHierarchyInfodataResponse(infores)
	detailListRsp := response.NewHierarchyDetailResponse(details)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, total)

	rsp := response.HierarchyReportResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		MetaDataResponse:     metadata,
		Data:                 hierachySummary,
		Summary:              countSummary,
		HierarchyInfo:        hierachyInfoRsp,
		DetailList:           detailListRsp,
	}
	handleSuccess(ctx, rsp)
}

type OfficeReportRequest struct {
	CadreName        string `json:"cadre_name"`
	DivisionOfficeId int64  `json:"division_office_id"`
	OfficeId         int64  `json:"office_id"`
	Search           string `json:"search"`
	IncludeList      bool   `json:"includelist"`
	port.MetaDataRequest
}

// GetOfficeReport godoc
//
// @Summary      Get Office Cadre Report
// @Description  Retrieves office-level cadre summaries, counts, hierarchy info, and optionally detailed post list.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        cadre_name         query     string  false  "Cadre name (optional filter)"
// @Param        division_office_id query     int     true   "Division Office ID (required)"
// @Param        office_id          query     int     false  "Office ID (optional, used when includelist=true)"
// @Param        search             query     string  false  "Search filter"
// @Param        includelist        query     bool    false  "Include detailed post list (true/false)"
// @Param        skip               query     int     false  "Number of records to skip for pagination"
// @Param        limit              query     int     false  "Maximum number of records to return"
// @Success      200  {object}  response.OfficeReportResponse "Office cadre report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/offices [get]
func (pmh *PostManagementHandler) GetOfficeReport(ctx *gin.Context) {
	var req OfficeReportRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	data, totalCount, err := pmh.svc.GetOffices(ctx, req.DivisionOfficeId, req.CadreName, &req.Search, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "DB error in GetOffices:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	info, err := pmh.svc.GetDivisionInfo(ctx, req.DivisionOfficeId)
	if err != nil {
		log.Error(ctx, "DB error in GetDivisionInfo:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	infoRsp := domain.OfficeInfo{
		Level:            4,
		LevelName:        "Final Offices",
		CadreName:        req.CadreName,
		DivisionOfficeId: req.DivisionOfficeId,
		DivisionName:     info.DivisionName,
		RegionOfficeID:   info.RegionOfficeID,
		RegionName:       info.RegionName,
		CircleOfficeID:   info.CircleOfficeID,
		CircleName:       info.CadreName,
	}

	summary := map[string]int{
		"totalPosts":  0,
		"totalFilled": 0,
		"totalVacant": 0,
	}
	for _, item := range data {
		summary["totalPosts"] += item.TotalPosts
		summary["totalFilled"] += item.TotalFilledPosts
		summary["totalVacant"] += item.TotalVacantPosts
	}

	var detailList []domain.OfficePostDetail
	if req.IncludeList && req.OfficeId != 0 {
		detailList, err = pmh.svc.DetailOfficeReport(ctx, req.CadreName, req.DivisionOfficeId, req.OfficeId, &req.Search)
		if err != nil {
			log.Error(ctx, "DB error in DetailOfficeReport:", err)
			apierrors.HandleDBError(ctx, err)
			return
		}
	}

	officeSummary := response.NewOfficeSummaryResponse(data)
	countSummary := response.NewCountSummaryResponse(summary["totalPosts"], summary["totalFilled"], summary["totalVacant"])
	officeInfoRsp := response.NewOfficeInfodataResponse(infoRsp)
	detailListRsp := response.NewOfficeDetailResponse(detailList)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, totalCount)

	rsp := response.OfficeReportResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		MetaDataResponse:     metadata,
		Data:                 officeSummary,
		Summary:              countSummary,
		HierarchyInfo:        officeInfoRsp,
		DetailList:           detailListRsp,
	}
	handleSuccess(ctx, rsp)
}

// GetPostReport godoc
//
// @Summary      Get Post Report
// @Description  Retrieves posts for a given cadre at different hierarchy levels (circle, region, division, office, or individual posts).
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        cadre_id           query     int     true   "Cadre ID (required)"
// @Param        office_id          query     int     false  "Filter by specific Office ID"
// @Param        circle_office_id   query     int     false  "Filter by Circle Office ID"
// @Param        region_office_id   query     int     false  "Filter by Region Office ID"
// @Param        division_office_id query     int     false  "Filter by Division Office ID"
// @Param        search             query     string  false  "Search filter for post details"
// @Param        includeList        query     bool    false  "Include detailed post list (true/false)"
// @Success      200  {object}  response.GetPostsResponse "Posts report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/posts [get]
func (pmh *PostManagementHandler) GetPostReport(ctx *gin.Context) {
	var req domain.PostReport

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	posts, err := pmh.svc.FetchPosts(ctx, req)
	if err != nil {
		log.Error(ctx, "DB error in FetchPosts:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	summary := domain.PostSummary{
		TotalPosts:  len(posts),
		TotalFilled: pmh.svc.CountStatus(posts, "Filled"),
		TotalVacant: pmh.svc.CountStatus(posts, "Vacant"),
	}

	var context domain.ContextInfo
	if req.OfficeID != nil {
		context, err = pmh.svc.GetContextInfo(ctx, *req.OfficeID)
		if err != nil {
			log.Error(ctx, "DB error in GetContextInfo:", err)
			apierrors.HandleDBError(ctx, err)
			return
		}
	}

	level := 4
	levelName := "All Posts for Cadre"
	if req.OfficeID != nil {
		level = 5
		levelName = "Individual Posts"
	} else if req.DivisionOfficeID != nil {
		levelName = "Posts in Division"
	} else if req.RegionOfficeID != nil {
		levelName = "Posts in Region"
	} else if req.CircleOfficeID != nil {
		levelName = "Posts in Circle"
	}

	hierarchy := domain.HierarchyPost{
		Level:            level,
		LevelName:        levelName,
		CadreID:          req.CadreID,
		OfficeID:         req.OfficeID,
		DivisionOfficeID: req.DivisionOfficeID,
		RegionOfficeID:   req.RegionOfficeID,
		CircleOfficeID:   req.CircleOfficeID,
	}

	officeSummary := response.NewPostSummaryResponse(posts)
	countSummary := response.NewPostSummaryResponse1(summary)
	contextRsp := response.NewContextInfoResponse(context)
	hierarchyRsp := response.NewHierarchyPost(hierarchy)

	rsp := response.GetPostsResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		DetailList:           officeSummary,
		Summary:              countSummary,
		HierarchyInfo:        hierarchyRsp,
		ContextInfo:          contextRsp,
	}
	handleSuccess(ctx, rsp)
}

type RegionRequest struct {
	CadreName            string `json:"cadre_name"`
	CircleOfficeId       int64  `json:"circle_office_id"`
	RegionOfficeId       int64  `json:"region_office_id"`
	Search               string `json:"search"`
	IncludeList          bool   `json:"include_list"`
	port.MetaDataRequest ``
}

// GetRegionHandler godoc
//
// @Summary      Get Region Report
// @Description  Retrieves region-level report for a given circle and cadre, with optional region details.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        cadre_name         query     string  false  "Cadre name to filter (optional)"
// @Param        circle_office_id   query     int     true   "Circle Office ID (required)"
// @Param        region_office_id   query     int     false  "Region Office ID (optional, used when IncludeList=true)"
// @Param        search             query     string  false  "Search filter for regions"
// @Param        include_list       query     bool    false  "Include detailed region list (true/false)"
// @Param        skip               query     int     false  "Pagination skip (default: 0)"
// @Param        limit              query     int     false  "Pagination limit (default: 10)"
// @Success      200  {object}  response.RegionReportResponse "Region report fetched successfully"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/regions [get]
func (pmh *PostManagementHandler) GetRegionHandler(ctx *gin.Context) {
	var req RegionRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	request := domain.RegionRequest{
		CadreName:      req.CadreName,
		CircleOfficeID: req.CircleOfficeId,
		RegionOfficeID: req.RegionOfficeId,
		Search:         req.Search,
		IncludeList:    req.IncludeList,
	}

	summaries, totalCount, err := pmh.svc.FetchRegionSummary(ctx, request, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "DB error in FetchRegionSummary:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	var details []domain.RegionDetail
	if req.IncludeList && req.RegionOfficeId > 0 {
		details, err = pmh.svc.FetchRegionDetails(ctx, request)
		if err != nil {
			log.Error(ctx, "DB error in FetchRegionDetails:", err)
			apierrors.HandleDBError(ctx, err)
			return
		}
	}

	circleInfo, err := pmh.svc.GetCircleInfo(ctx, req.CircleOfficeId)
	if err != nil {
		log.Error(ctx, "DB error in GetCircleInfo:", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	hierarchyInfo := domain.HierarchyRegion{
		Level:          2,
		LevelName:      "Regions",
		CadreName:      req.CadreName,
		CircleOfficeID: req.CircleOfficeId,
		CircleName:     circleInfo.CircleName,
	}

	officeSummary := response.NewRegionSummaryResponse(summaries)
	countSummary := response.NewRegionCountResponse(summaries)
	officeInfoRsp := response.NewHierarchyRegionResponse(hierarchyInfo)
	detailListRsp := response.NewRegionDetailResponse(details)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, totalCount)

	rsp := response.RegionReportResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		MetaDataResponse:     metadata,
		Data:                 officeSummary,
		Summary:              countSummary,
		HierarchyInfo:        officeInfoRsp,
		DetailList:           detailListRsp,
	}
	handleSuccess(ctx, rsp)
}

type ListCadreWiseReportRequest struct {
	DivisionID int `uri:"division-id" validate:"required"`
	CadreID    int `uri:"cadre-id" validate:"required"`
}

// GetListCadreWiseOfficeWiseReportsHandler godoc
//
// @Summary      Get Cadre-wise Office-wise Reports
// @Description  Retrieves all office-wise post availability details for a given cadre within a division.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param        division-id   path      int  true  "Division ID"
// @Param        cadre-id      path      int  true  "Cadre ID"
// @Success      200  {object}  response.ListCadreWiseOfficeAPIResponse "List of office-wise reports for a cadre"
// @Failure      400  {object}  apierrors.APIErrorResponse "Invalid request parameters"
// @Failure      500  {object}  apierrors.APIErrorResponse "Internal server error"
// @Router       /reports/list-cadre-wise-reports/{division-id}/{cadre-id} [get]
func (pmh *PostManagementHandler) GetListCadreWiseOfficeWiseReportsHandler(ctx *gin.Context) {
	var req ListCadreWiseReportRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetListCadreWiseOfficeWiseReports(ctx, req.DivisionID, req.CadreID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Cadre wise reports:", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewListCadreWiseOfficeWiseReportsResponse(postsavbl)

	apiRsp := response.ListCadreWiseOfficeAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetListCadreWiseOfficeWiseReports API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// CheckPostsStatusHandler to check the posts filled and vacant status before deleting the office.
// @Summary      fetches the details of posts filled and vacant status based on office id.
// @Description  fetches the details of posts filled and vacant status based on office id.
// @Tags         Posts
// @Accept       json
// @Produce      json
// @Param office-id query string true "Office ID"
// @Success 200 {object} response.GetPostsFilledVacantStatusDetailedAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /posts/check-posts-status [get]
func (pmh *PostManagementHandler) CheckPostsStatusHandler(ctx *gin.Context) {
	var req CheckPostsStatusRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	posts, err := pmh.svc.GetPostsFilledVacantStatusDetailed(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching post details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	var message string
	var filledPosts, unfilledPosts []domain.PostsFilledVacantStatusDetailed

	if len(posts) == 0 {
		message = "No posts are available at the office. You may proceed with deletion."
	} else {
		for _, post := range posts {
			if post.EmployeeID.Valid && post.EmployeeID.Int != 0 {
				filledPosts = append(filledPosts, post)
			} else {
				unfilledPosts = append(unfilledPosts, post)
			}
		}

		switch {
		case len(filledPosts) > 0:
			message = "Filled posts are available. Please delink the employees and redeploy the posts before deleting the office."
		case len(unfilledPosts) > 0:
			message = "Only unfilled posts are available. Please redeploy the posts before deleting the office."
		default:
			message = "No posts are filled or vacant. You may proceed with deletion."
		}
	}

	apiRsp := response.GetPostsFilledVacantStatusDetailedAPIResponse{
		StatusCode: 200,
		Message:    message,
		Data:       response.NewGetPostsFilledVacantStatusDetailedResponse(posts),
		TotalCount: len(posts),
	}

	log.Debug(ctx, "CheckPostsStatusHandler API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type CheckPostsStatusRequest struct {
	OfficeID int `form:"office-id" validate:"required,office_id"`
}

// GetPostAuthorityChargesDetailsHandler to check authority charges details based on office_id.
// @Summary      fetches the authority charges for a office.
// @Description  retrieves the authority charges for a office.
// @Tags         Reports
// @Accept       json
// @Produce      json
// @Param office-id path int true "Office ID"
// @Success 200 {object} response.PostAuthorityDetailsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router /reports/post-authority-details/{office-id} [get]
func (pmh *PostManagementHandler) GetPostAuthorityChargesDetailsHandler(ctx *gin.Context) {
	var req GetPostAuthorityChargesDetailsRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostAuthorityChargesDetailByOfficeRepo(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching reports", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	if len(postsavbl) == 0 {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "No data found in database")
		return
	}
	rsp := response.NewPostAuthorityDetailsResponse(postsavbl)

	apiRsp := response.PostAuthorityDetailsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "PostAuthorityDetails API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostAuthorityChargesDetailsRequest struct {
	OfficeID int64 `uri:"office-id" validate:"required"`
}

// DeletePostsbyOfficeIDHandler to delete posts by office Id.
// @Summary      deletes the posts for a office.
// @Description  deletes the posts for a office.
// @Tags        posts
// @Accept       json
// @Produce      json
// @Param office-id query int true "Office ID"
// @Param user-id query string true "User ID performing deletion"
// @Success 200 {object} response.DeletePostsbyOfficeIDAPIResponse "Success -Successfully deleted Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router /posts/delete-by-office-id [delete]
func (pmh *PostManagementHandler) DeletePostsbyOfficeIDHandler(ctx *gin.Context) {
	var req DeletePostsbyOfficeIDRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}
	// Fetch posts for the office
	posts, err := pmh.svc.GetPostsFilledVacantStatusDetailed(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching posts", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	if len(posts) == 0 {
		apierrors.HandleValidationError(ctx, fmt.Errorf("no posts found for the given office ID"))
		return
	}

	// Check if all posts are vacant
	allVacant := true
	for _, post := range posts {
		if post.EmployeeName.Valid && post.EmployeeName.String != "Vacant" {
			allVacant = false
			break
		}
	}

	if !allVacant {
		apierrors.HandleValidationError(ctx, fmt.Errorf("cannot delete posts: some posts are filled"))
		return
	}
	// delete posts
	message, err := pmh.svc.DeletePostsbyOfficeIDRepo(ctx, req.OfficeID, req.UserID)
	if err != nil {
		log.Error(ctx, "Database error while deleting posts", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.DeletePostsbyOfficeIDAPIResponse{
		StatusCodeAndMessage: port.StatusCodeAndMessage{
			StatusCode: http.StatusOK,
			Success:    true,
			Message:    "DeletePostsbyOfficeID success",
		},
		Data: response.DeletePostsbyOfficeIDResponse{
			OfficeID: req.OfficeID,
			Message:  message,
			Deleted:  len(posts),
		},
	}
	log.Debug(ctx, "DeletePostsbyOfficeID API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type DeletePostsbyOfficeIDRequest struct {
	OfficeID int    `form:"office-id" validate:"required"`
	UserID   string `form:"user-id" validate:"required"`
}

// CheckCadreExistsHandler to check if the given post id is or not PS Group B Cadre
// @Summary      checks if the given post id is or not PS Group B Cadre
// @Description  checks if the given post id is or not PS Group B Cadre
// @Tags        posts
// @Accept       json
// @Produce      json
// @Param post-id query int true "Post ID"
// @Success 200 {object} response.CheckCadresAPIResponse "Success -Successfully checked Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router /posts/check-cadres [get]
func (pmh *PostManagementHandler) CheckCadreExistsHandler(ctx *gin.Context) {
	var req CheckCadreExistsRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}
	//check the cadre and give bool response
	exists, err := pmh.svc.CheckCadreExistsRepo(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "Database error while fetching posts", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	// Build response using the response struct
	resp := response.CheckCadresAPIResponse{
		Data: response.CheckCadreData{
			CadreExists: exists,
		},
	}

	// Return JSON response
	ctx.JSON(http.StatusOK, resp)
}

type CheckCadreExistsRequest struct {
	PostID int `form:"post-id" validate:"required"`
}

type GetSanctionedStrengthByOfficeIDRequest struct {
	OfficeID int64 `form:"office-id" validate:"required"`
	port.MetaDataRequest
}

// GetSanctionedStrengthByOfficeIDHandler godoc
//
// @Summary      Get Sanctioned Strength by Office ID
// @Description  Retrieve sanctioned strength details for a given office ID.
// @Tags         Post Management
// @Accept       json
// @Produce      json
//
// @Param        office-id   query     int    true   "Office ID"
// @Param        skip        query     int    false  "Pagination skip value"
// @Param        limit       query     int    false  "Pagination limit value"
//
// @Success      200  {object}  response.ListSanctionedStrengthAPIResponse   "Successful response with sanctioned strength details"
// @Failure      400  {object}  apierrors.APIErrorResponse                  "Validation or binding error"
// @Failure      404  {object}  apierrors.APIErrorResponse                  "Data not found"
// @Failure      500  {object}  apierrors.APIErrorResponse                  "Internal server error"
//
// @Router       /posts/sanctioned-strength [get]
func (pmh *PostManagementHandler) GetSanctionedStrengthByOfficeIDHandler(ctx *gin.Context) {
	var req GetSanctionedStrengthByOfficeIDRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	strength, err := pmh.svc.GetSanctionedStrengthByOfficeIDRepo(ctx, req.OfficeID, req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "Database error while fetching sanctioned strength", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewSanctionedStrengthResponse(strength)
	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(strength))

	apiRsp := response.ListSanctionedStrengthAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetSanctionedStrengthByOfficeID API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type UpdatePostNameByIdRequest struct {
	PostID    int    `form:"post_id" validate:"required"`
	PostName  string `form:"post_name" validate:"required"`
	Remarks   string `form:"remarks" validate:"required"`
	UpdatedBy string `form:"updated_by" validate:"required"`
}

func (pmh *PostManagementHandler) UpdatePostnameByPostIdByHandler(ctx *gin.Context) {

	var req UpdatePostNameByIdRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	postsavbl, err := pmh.svc.GetPostDetailsbyPostID(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "error while fetching post details", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}
	if postsavbl == nil {
		log.Error(ctx, "post id not found", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}

	_, err = pmh.svc.UpdatePostNamebyPostIDRepo(ctx, req.PostID, req.PostName, req.Remarks, req.UpdatedBy)
	if err != nil {
		log.Error(ctx, "Database error while post updation", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.UpdatePostnameByIdResponse{
		StatusCodeAndMessage: port.UpdateSuccess,
		Data:                 "Post details updated successfully",
	}

	log.Debug(ctx, "UpdatePostDetailsbyPostID API response:  %s", apiRsp)
	handleCreateSuccess(ctx, apiRsp)
}

// GetPostDetailsByPostIdHandler godoc
// @Summary     Get post details by Post ID
// @Description Fetch detailed post information using the unique Post ID.
// @Tags        Post Management
// @Accept      json
// @Produce     json
//
// @Param       post-id   path     int  true  "Post ID"
//
// @Success     200 {object} response.GetPostDetailsbyPostIDAPIResponse1 "Post details fetched successfully"
// @Failure     400 {object} apierrors.APIErrorResponse                  "Binding or validation error"
// @Failure     404 {object} apierrors.APIErrorResponse                  "Post not found"
// @Failure     500 {object} apierrors.APIErrorResponse                  "Internal server error"
//
// @Router      /post-management/posts/get-post/{post-id} [get]
func (pmh *PostManagementHandler) GetPostDetailsByPostIdHandler(ctx *gin.Context) {
	var req GetPostDetailsbyPostIDRequest
	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostDetailsbyPostID1(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "Database error while fetching post details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	rsp := response.NewGetPostDetailsbyPostIDAPIResponse(postsavbl)

	apiRsp := response.GetPostDetailsbyPostIDAPIResponse1{
		StatusCodeAndMessage: port.FetchSuccess,
		Data:                 rsp,
	}

	log.Debug(ctx, "GetPostDetailsbyPostID API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type PostManagementSummaryRequest struct {
	CadreName      string `form:"cadre_name"`
	CircleOfficeId int64  `form:"circle_office_id"`
	IncludeList    bool   `form:"include_list"`
	Search         string `form:"search"`
	port.MetaDataRequest
}

func (pmd *PostManagementHandler) GetPostManagemmentSummaryHandler(ctx *gin.Context) {
	var req PostManagementSummaryRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	resp, err := pmd.svc.GetPostSummaryRepo(
		ctx,
		req.CadreName,
		req.CircleOfficeId,
		req.IncludeList,
		req.MetaDataRequest,
	)
	if err != nil {
		log.Error(ctx, "Error while fetching post summary", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.GetPostManagementSummaryAPIResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		Data: response.CaddreSummaryResponse{
			Total: resp.Total,
		},
	}

	// includeList = false → summary
	if !req.IncludeList {
		apiRsp.Data.Summary = resp.Summary
	}

	// includeList = true → detailed list
	if req.IncludeList {
		apiRsp.Data.List = resp.List
	}

	handleSuccess(ctx, apiRsp)
}

type CircleSummaryRequest struct {
	CadreName      string `form:"cadre_name"`
	CircleOfficeID int64  `form:"circle_office_id"`
	RegionOfficeID int64  `form:"region_office_id"`
	IncludeList    bool   `form:"include_list"`
	Search         string `form:"search"`
	port.MetaDataRequest
}

func (pmd *PostManagementHandler) GetCircleSummaryHandler(ctx *gin.Context) {
	var req CircleSummaryRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	resp, err := pmd.svc.GetCircleSummaryRepo(
		ctx,
		req.CadreName,
		req.CircleOfficeID,
		req.RegionOfficeID,
		req.IncludeList,
		req.MetaDataRequest,
	)
	if err != nil {
		log.Error(ctx, "Error while fetching circle summary", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.GetPostManagementCircleSummaryAPIResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		Data: response.CircleSummaryResponse1{
			Total: resp.Total,
		},
	}

	// includeList = false → summary
	if !req.IncludeList {
		apiRsp.Data.Summary = resp.Summary
		apiRsp.Data.Hierarchy = resp.Hierarchy
	}

	// includeList = true → detailed list
	if req.IncludeList {
		apiRsp.Data.List = resp.List
	}

	handleSuccess(ctx, apiRsp)
}

type RegionSummaryRequest struct {
	CadreName        string `form:"cadre_name"`
	DivisionOfficeID int64  `form:"division_office_id"`
	RegionOfficeID   int64  `form:"region_office_id"`
	IncludeList      bool   `form:"include_list"`
	port.MetaDataRequest
}

func (pmd *PostManagementHandler) GetRegionSummaryHandler(ctx *gin.Context) {
	var req RegionSummaryRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	resp, err := pmd.svc.GetRegionSummaryRepo(
		ctx,
		req.CadreName,
		req.DivisionOfficeID,
		req.RegionOfficeID,
		req.IncludeList,
		req.MetaDataRequest,
	)
	if err != nil {
		log.Error(ctx, "Error while fetching region summary", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.GetPostManagementRegionSummaryAPIResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		Data: response.RegionSummaryResponse1{
			Total: resp.Total,
		},
	}

	// includeList = false → region summary
	if !req.IncludeList {
		apiRsp.Data.Summary = resp.Summary
		apiRsp.Data.Hierarchy = resp.Hierarchy
	}

	// includeList = true → detailed list
	if req.IncludeList {
		apiRsp.Data.List = resp.List
	}

	handleSuccess(ctx, apiRsp)
}

type DivisionSummaryRequest struct {
	CadreName        string `form:"cadre_name"`
	OfficeID         int64  `form:"office_id"`
	DivisionOfficeID int64  `form:"division_office_id"`
	IncludeList      bool   `form:"include_list"`
	port.MetaDataRequest
}

func (pmd *PostManagementHandler) GetDivisionSummaryHandler(ctx *gin.Context) {
	var req DivisionSummaryRequest

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	resp, err := pmd.svc.GetDivisionSummaryRepo(
		ctx,
		req.CadreName,
		req.OfficeID,
		req.DivisionOfficeID,
		req.IncludeList,
		req.MetaDataRequest,
	)
	if err != nil {
		log.Error(ctx, "Error while fetching division summary", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.GetPostManagementDivisionSummaryAPIResponse{
		StatusCodeAndMessage: port.FetchSuccess,
		Data: response.DivisionSummaryResponse1{
			Total: resp.Total,
		},
	}

	// includeList = false → division summary
	if !req.IncludeList {
		apiRsp.Data.Summary = resp.Summary
		apiRsp.Data.Hierarchy = resp.Hierarchy
	}

	// includeList = true → detailed list
	if req.IncludeList {
		apiRsp.Data.List = resp.List
	}

	handleSuccess(ctx, apiRsp)
}

type PostManagementSummaryRequest1 struct {
	CadreName   string `form:"cadre_name"`
	IncludeList bool   `form:"include_list"`
	Search      string `form:"search"`
	port.MetaDataRequest
}

func (pmd *PostManagementHandler) GetPostManagementSummaryHandler(ctx *gin.Context) {
	var req PostManagementSummaryRequest1

	if err := apiutility.BindAndValidate(ctx, &req, false, true, false); err != nil {
		log.Error(ctx, err)
		return
	}

	resp, err := pmd.svc.GetPostSummaryRepo1(
		ctx,
		req.CadreName,
		req.IncludeList,
		req.Search,
		req.MetaDataRequest,
	)
	if err != nil {
		log.Error(ctx, "Error while fetching post management summary", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.GetPostManagementDivisionSummaryAPIResponse1{
		StatusCodeAndMessage: port.FetchSuccess,
		Data: response.PostSummaryResponse1{
			Total: resp.Total,
		},
	}

	// include_list = false → summary
	if !req.IncludeList {
		apiRsp.Data.Summary = resp.Summary
	}

	// include_list = true → detailed list
	if req.IncludeList {
		apiRsp.Data.List = resp.List
	}

	handleSuccess(ctx, apiRsp)
}
