package handler

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"path"
	"path/filepath"
	"pmdm/core/domain"
	"pmdm/core/port"
	"pmdm/handler/response"
	repo "pmdm/repo/postgres"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
	apiutility "gitlab.cept.gov.in/it-2.0-common/api-utility"
	validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
)

type PosttoPostMappingrHandler struct {
	svc *repo.PosttoPostMappingRepository
}

// NewCadreMasterHandler creates a new CadreMasterHandler instance
func NewPosttoPostMappingrHandler(svc *repo.PosttoPostMappingRepository) *PosttoPostMappingrHandler {
	return &PosttoPostMappingrHandler{
		svc,
	}
}

type ListMulAuthRequest struct {
	PostID int    `uri:"post-id" validate:"required"`
	RoleId string `uri:"role-id" validate:"required"`
	port.MetaDataRequest
}

// GetMulAuthID godoc
//
//	@Summary		Get Multiple Authorities by Post ID and Role IDs
//	@Description	Fetches the multiple authority mappings for a given post ID and list of role IDs.
//	@Tags			Post Management
//	@Accept			json
//	@Produce		json
//	@Param			post-id	path		int	true	"Post ID"
//	@Param			role-id	path		string	true	"Comma-separated list of Role IDs"
//	@Param       MetaDataRequest	query		port.MetaDataRequest	false  		"Metadata request containing skip, limit, orderBy & sortBy"
//	@Success		200		{object}	response.GetMulAuthIDAPIResponse	"Multiple authority mappings retrieved successfully"
//	@Failure		400		{object}	apierrors.APIErrorResponse		"Validation error"
//	@Failure		401		{object}	apierrors.APIErrorResponse		"Unauthorized error"
//	@Failure		403		{object}	apierrors.APIErrorResponse		"Forbidden error"
//	@Failure		404		{object}	apierrors.APIErrorResponse		"Data not found error"
//	@Failure		500		{object}	apierrors.APIErrorResponse		"Internal server error"
//	@Router			/post-management/ptop-mappings/posts/{post-id}/roles/{role-id} [get]
func (pph *PosttoPostMappingrHandler) GetMulAuthIDHandler(ctx *gin.Context) {
	var req ListMulAuthRequest
	if err := ctx.ShouldBindUri(&req); err != nil {
		log.Error(ctx, "Binding failed for ListMulAuthRequest: %s", err.Error())
		apierrors.HandleBindingError(ctx, err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		log.Error(ctx, "Validation failed for ListMulAuthRequest: %s", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}

	if req.Limit == 0 {
		req.Limit = math.MaxInt32
	}

	// Fetch multiple authorities based on Post ID and Role IDs
	authrows, err := pph.svc.GetMulAuthPostID(ctx, req.PostID, strings.Split(req.RoleId, ","), req.MetaDataRequest)
	if err != nil {
		log.Error(ctx, "GetMulAuthPostID repo call failed: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Prepare the response data
	rsps := []response.GetMulAuthIDResponse{}
	for _, authrow := range authrows {
		rsps = append(rsps, response.NewGetMulAuthIDResponse(authrow))
	}

	metadata := port.NewMetaDataResponse(req.Skip, req.Limit, len(rsps))
	// Directly create the response
	apiRsp := response.GetMulAuthIDAPIResponse{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		TotalRecords:         len(rsps),
		Data:                 rsps,
	}

	// Send the success response
	handleSuccess(ctx, apiRsp)
}

// PostMappingDetailUpdateRequest represents a request body for updating post mapping details
type PostMappingDetailUpdateRequest struct {
	EmpPostID  int           `json:"employee_post_id" validate:"required"`
	FieldName  []string      `json:"field_name" validate:"required"`
	FieldValue []interface{} `json:"field_value" validate:"required"`
}

type PostMappingDetailCreateRequest struct {
	EmpPostID   int `json:"employee_post_id" validate:"required"`
	EmpOfficeID int `json:"employee_office_id" validate:"required"`
}

type UpdateArrayOfEmpPostIDForParticularFieldRequest struct {
	OfficeID   int         `json:"office_id" validate:"required"`
	EmpPostID  []int       `json:"employee_post_id" validate:"required"`
	FieldName  string      `json:"field_name" validate:"required"`
	FieldValue interface{} `json:"field_value" validate:"required"`
}

// ListPostManagementMaster godoc

//	@Summary        Multiple Employee Post ID updated for  Particular FieldName with FieldValue
//	@Description    Multiple Employee Post ID updated for  Particular FieldName with FieldValue by giving requisite details
//	@Tags           PostToPostMapping
//	@Accept         json
//	@Produce        json
//
// @Param        request  body  UpdateArrayOfEmpPostIDForParticularFieldRequest  true  "Update payload with office ID, employee post IDs, field name, and value"
//
// " Multiple Employee Post ID updated for Particular FieldName with FieldValue by giving requisite details"
//
//	@Success        200                     {object}    PostMapUpdateResponseArray       "UpdatePostMappingDetail"
//	@Failure        400                     {object}    apierrors.APIErrorResponse          "Validation error"
//	@Failure        401                     {object}    apierrors.APIErrorResponse          "Unauthorized error"
//	@Failure        403                     {object}    apierrors.APIErrorResponse          "Forbidden error"
//	@Failure        404                     {object}    apierrors.APIErrorResponse          "Data not found error"
//	@Failure        500                     {object}    apierrors.APIErrorResponse          "Internal server error"
//	@Router         /ptopmap/updatemultipleemppostid [put]
//
// Function to update a specific field in the post mapping details
func (pmh *PosttoPostMappingrHandler) UpdateArrayOfEmpPostIDForParticularField1(ctx *gin.Context) {
	var req UpdateArrayOfEmpPostIDForParticularFieldRequest
	if err := ctx.ShouldBindJSON(&req); err != nil {
		validationError(ctx, err)
		return
	}

	// Validate the request
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	// Pass all EmpPostID to the repository along with the field value
	updateResponse, err := pmh.svc.UpdateArrayOfEmpPostIDForParticularFieldQuery(ctx, req.EmpPostID, req.FieldName, req.FieldValue, req.OfficeID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Create the field name and field value for the response
	fieldName := req.FieldName
	fieldValue := req.FieldValue

	// Create the response array with field name and field value included
	rsp := NewPostMapUpdateResponseArray(updateResponse, fieldName, fieldValue)
	handleSuccess(ctx, rsp)
}

type getAuthforMultiple struct {
	PostIDs []int `form:"post_id" validate:"required"`
}

func (pph *PosttoPostMappingrHandler) GetAuthorityDetailsForMultiplePostID1(ctx *gin.Context) {
	var req getAuthforMultiple
	if err := ctx.ShouldBindQuery(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Debug(ctx, "error %s", err)
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

type UpdateArrayOfEmpPostIDForManyFieldRequest struct {
	OfficeID   []int         `json:"office_id" validate:"required"`
	EmpPostID  []int         `json:"employee_post_id" validate:"required"`
	FieldName  []string      `json:"field_name" validate:"required"`
	FieldValue []interface{} `json:"field_value" validate:"required"`
}

type MultipleEmpPostIDForMultipleFieldRequest struct {
	OfficeID   int           `json:"office_id" validate:"required"`
	EmpPostID  []int         `json:"employee_post_id" validate:"required"`
	FieldName  []string      `json:"field_name" validate:"required"`
	FieldValue []interface{} `json:"field_value" validate:"required"`
	PostID     string        `json:"approve_post_id" validate:"required"`
	CreatedBy  string        `json:"created_by" validate:"required"`
}

// func (pmh *PosttoPostMappingrHandler) CreatePostMappingDetailMaker1(ctx *gin.Context) {
// 	var req MultipleEmpPostIDForMultipleFieldRequest
// 	if err := ctx.ShouldBindJSON(&req); err != nil {
// 		validationError(ctx, err)
// 		return
// 	}
// 	// Validate the request
// 	if err := validation.ValidateStruct(req); err != nil {
// 		apierrors.HandleValidationError(ctx, err)
// 		return
// 	}
// 	fmt.Println("Value of approvepostid", req.PostID)
// 	// Pass all EmpPostID to the repository along with the field value
// 	updateResponse, err := pmh.svc.CreatePostMappingDetailMaker(ctx, req.EmpPostID, req.FieldName, req.FieldValue, req.OfficeID)
// 	if err != nil {
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}

// 	// Create the response array with field names and field values included
// 	rsp := NewPostMapUpdateResponseArrayForMultipleFields(updateResponse)
// 	handleSuccess(ctx, rsp)
// }

func (pph *PosttoPostMappingrHandler) GetPostMappingMasterMaker1(ctx *gin.Context) {
	var req ListAuthRequest
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Debug(ctx, "error %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		return
	}

	//rid := ctx.Query("role_id")
	authrows, err := pph.svc.GetPostMappingMasterMaker(ctx, req.PostID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		return
	}
	//rsp := newAuthorityDetailsResponse(authrows)
	handleSuccess(ctx, authrows)

}

type ApprovePostMappingDetailRequest struct {
	EmployeePostID int    `json:"employee_post_id" validate:"required"`
	ApprovedBy     string `json:"approved_by" validate:"required"`
	FieldName      string `json:"field_name" validate:"required" `
	FieldValue     int32  `json:"field_value" `
	Status         string `json:"approve_status"  `
	OfficeID       int    `json:"office_id" `
	Remarks        string `json:"remarks"  `
}

// type getMasterAuth struct {
// 	PostID int `uri:"post_id" validate:"required"`
// }

// GetPostRedeplomentByOfficeIDHandler retrieves the list of posts avaiable in the circle.
// @Summary      fetches the list of posts available in the circle.
// @Description  Retrieves the list of posts avaiable in the circle from a given office_id.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param office-id path int true "Office ID"
// @Param cadre-name path string true "Cadre Name"
// @Success 200 {object} response.GetPostRedeplomentByOfficeIDAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/post-redeployment/:office-id/:cadre-name [get]
func (pmh *PosttoPostMappingrHandler) GetPostRedeplomentByOfficeIDHandler(ctx *gin.Context) {
	var req GetPostRedeplomentByOfficeIDRepoRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}

	postsavbl, err := pmh.svc.GetPostRedeplomentByOfficeID(ctx, req.OfficeID, req.CadreName)
	if err != nil {
		log.Error(ctx, "Database error while fetching Post details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetPostRedeplomentByOfficeIDResponse(postsavbl)

	apiRsp := response.GetPostRedeplomentByOfficeIDAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostRedeplomentByOfficeID API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostRedeplomentByOfficeIDRepoRequest struct {
	OfficeID  int    `uri:"office-id" validate:"required"`
	CadreName string `uri:"cadre-name" validate:"required"`
}

// // SavePostRedeploymentHandler to save the post redeployment entries in the database.
// // @Summary      to save the post redeployment entries in the database.
// // @Description  to save the post redeployment entries in the database.
// // @Tags         Ptop-mappings
// // @Accept       json
// // @Produce      json
// // @Param request body SavePostRedeploymentRequest true "Save Post Redeployment Request"
// // @Success 201 {object} response.SavePostRedeploymentAPIResponse "Success - Post redeployed successfully"
// // @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// // @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// // @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// // @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// // @Router       /ptop-mappings/save-post-redployment [post]
// func (pmh *PosttoPostMappingrHandler) SavePostRedeploymentHandler(ctx *gin.Context) {
// 	var req SavePostRedeploymentRequest

// 	if err := apiutility.BindAndValidate(ctx, &req, false, false, true); err != nil {
// 		log.Error(ctx, err)
// 		return
// 	}

// 	// Check if the target office is compatible for the post
// 	compatible, err := pmh.svc.CheckOfficeCompatibility(ctx, req.PostID, req.RedeploymentToOfficeID)
// 	if err != nil {
// 		log.Error(ctx, "Redeployment to office is not compatible", err.Error())
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}
// 	if !compatible {
// 		ctx.JSON(400, gin.H{"error": "Target office is not compatible for redeployment"})
// 		return
// 	}

// 	// Map request to domain structs
// 	postRedeploymentLog := ToPostRedeploymentLog(req)
// 	postManagementMasterUpdate := ToPostManagementMasterUpdate(req)

// 	_, err = pmh.svc.SavePostRedeploymentRepo(ctx, postRedeploymentLog, postManagementMasterUpdate)
// 	if err != nil {
// 		log.Error(ctx, "Database error while post redeployment", err.Error())
// 		apierrors.HandleDBError(ctx, err)
// 		return
// 	}

// 	apiRsp := response.SavePostRedeploymentAPIResponse{
// 		StatusCodeAndMessage: port.CreateSuccess,
// 		Data:                 "Post redeployment recorded successfully",
// 	}

// 	log.Debug(ctx, "SavePostRedeploymentHandler response:  %s", apiRsp)
// 	handleCreateSuccess(ctx, apiRsp)
// }

// GetCircleOfficeIDsHandler retrieves the list of circle IDs and Circle Names.
// @Summary      fetches the list of Circle IDs and Circle Names.
// @Description  Retrieves the list of circle IDs and Circle Names.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Success 200 {object} response.GetCircleOfficeIDsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/circlenames [get]
func (pmh *PosttoPostMappingrHandler) GetCircleOfficeIDsHandler(ctx *gin.Context) {

	postsavbl, err := pmh.svc.GetCircleOfficeIDs(ctx)
	if err != nil {
		log.Error(ctx, "Database error while fetching circle office details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetCircleOfficeIDsResponse(postsavbl)

	apiRsp := response.GetCircleOfficeIDsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetCircleOfficeIDs API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// GetRegionalOfficeIDsHandler retrieves the list of Region IDs and Region Names.
// @Summary      fetches the list of Region IDs and Region Names.
// @Description  Retrieves the list of Region IDs and Region Names.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param circle-id path string true "circle ID"
// @Success 200 {object} response.GetRegionalOfficeIDsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/regionnames/:circle-id [get]
func (pmh *PosttoPostMappingrHandler) GetRegionalOfficeIDsHandler(ctx *gin.Context) {
	var req GetRegionalOfficeIDsRepoRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetRegionalOfficeIDs(ctx, req.CircleID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Regional office details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetRegionalOfficeIDsResponse(postsavbl)

	apiRsp := response.GetRegionalOfficeIDsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetRegionalOfficeIDs API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetRegionalOfficeIDsRepoRequest struct {
	CircleID int `uri:"circle-id" validate:"required"`
}

// GetDivisionalOfficeIDsHandler retrieves the list of Division IDs and Division Names based on Region ID.
// @Summary      fetches the list of Division IDs and Division Names based on Region ID.
// @Description  Retrieves the list of Division IDs and Division Names based on Region ID.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param region-id path string true "region ID"
// @Success 200 {object} response.GetDivisionalOfficeIDsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/divisionnames/:region-id [get]
func (pmh *PosttoPostMappingrHandler) GetDivisionalOfficeIDsHandler(ctx *gin.Context) {
	var req GetDivisionalOfficeIDsRepoRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetDivisionalOfficeIDs(ctx, req.RegionID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Divisional office details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetDivisionalOfficeIDsResponse(postsavbl)

	apiRsp := response.GetDivisionalOfficeIDsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetDivisionalOfficeIDs API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetDivisionalOfficeIDsRepoRequest struct {
	RegionID int `uri:"region-id" validate:"required"`
}

// GetCadreDetailsHandler retrieves the list of cadre IDs and cadre Names.
// @Summary      fetches the list of Cadre IDs and cadre Names.
// @Description  Retrieves the list of Cadre IDs and cadre Names.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Success 200 {object} response.GetCadreDetailsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/cadrenames [get]
func (pmh *PosttoPostMappingrHandler) GetCadreDetailsHandler(ctx *gin.Context) {

	postsavbl, err := pmh.svc.GetCadreDetails(ctx)
	if err != nil {
		log.Error(ctx, "Database error while fetching Cadre details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetCadreDetailsResponse(postsavbl)

	apiRsp := response.GetCadreDetailsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetCadreDetails API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// GetPostAndEmployeeHierarchyHandler retrieves the list of Posts and Employee Hierarchy based on Office ID.
// @Summary      fetches the list of Posts and Employee Hierarchy based on Office ID.
// @Description  Retrieves the list of Posts and Employee Hierarchy based on Office ID.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param office-id path string true "Office ID"
// @Success 200 {object} response.GetPostAndEmployeeHierarchyAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings-makers/hierarchy-posts/:office-id [get]
func (pph *PosttoPostMappingrHandler) GetPostAndEmployeeHierarchyHandler(ctx *gin.Context) {
	var req GetPostAndEmployeeHierarchRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	officelist, err := pph.svc.GetPostAndEmployeeHierarchy(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching Divisional office details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetPostAndEmployeeHierarchyResponse(officelist)

	apiRsp := response.GetPostAndEmployeeHierarchyAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostAndEmployeeHierarchy API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostAndEmployeeHierarchRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

// GetPostDetailsForRedeploymentHandler retrieves the post id details based on PostID for redeployment.
// @Summary      fetches the post id details based on Post ID for redeployment.
// @Description  Retrieves the post id details based on Post ID for redeployment.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param post-id path string true "post ID"
// @Success 200 {object} response.GetPostDetailsForRedeploymentAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/post-details/{post-id} [get]
func (pmh *PosttoPostMappingrHandler) GetPostDetailsForRedeploymentHandler(ctx *gin.Context) {
	var req GetPostDetailsForRedeploymentRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostDetailsForRedeployment(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "Database error while fetching post details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetPostDetailsForRedeploymentResponse(postsavbl)

	apiRsp := response.GetPostDetailsForRedeploymentAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostDetailsForRedeployment API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostDetailsForRedeploymentRequest struct {
	PostID int `uri:"post-id" validate:"required"`
}

// GetPostDetailsHandler retrieves the list of Post IDs and Post Names.
// @Summary      fetches the list of Post IDs and Post Names.
// @Description  Retrieves the list of Post IDs and Post Names.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Success 200 {object} response.GetPostDetailsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/postnames [get]
func (pmh *PosttoPostMappingrHandler) GetPostDetailsHandler(ctx *gin.Context) {

	postsavbl, err := pmh.svc.GetPostDetails(ctx)
	if err != nil {
		log.Error(ctx, "Database error while fetching Posts details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetPostDetailsResponse(postsavbl)

	apiRsp := response.GetPostDetailsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostDetails API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// GetDesignationDetailsHandler retrieves the list of Designation IDs and Designation..
// @Summary      fetches the list of Designation IDs and Designation.
// @Description  Retrieves the list of Designation IDs and Designation.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Success 200 {object} response.GetDesignationDetailsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/designationnames [get]
func (pmh *PosttoPostMappingrHandler) GetDesignationDetailsHandler(ctx *gin.Context) {

	postsavbl, err := pmh.svc.GetDesignationDetails2(ctx)
	if err != nil {
		log.Error(ctx, "Database error while fetching Designation details", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetDesignaationDetailsResponse(postsavbl)

	apiRsp := response.GetDesignationDetailsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetDesignationDetails API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

// GetPostRedeployedInwardReportsHandler to fetch the report on redeployed posts to an office by circle office id.
// @Summary      fetches the report on redeployed posts to an office by circle office id.
// @Description  retrieves the report on redeployed posts to an office by circle office id.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param office-id path string true "Office ID"
// @Success 200 {object} response.GetPostRedeployedInwardReportsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/post-redeployed-inward-reports/:office-id [get]
func (pmh *PosttoPostMappingrHandler) GetPostRedeployedInwardReportsHandler(ctx *gin.Context) {
	var req GetPostRedeployedInwardReportsRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostRedeployedInwardReports(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching reports", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	if len(postsavbl) == 0 {
		log.Error(ctx, "No data found in database")
		apierrors.HandleDBError(ctx, errors.New("No data found in database"))
		return
	}

	rsp := response.NewGetPostRedeployedInwardReportsResponse(postsavbl)

	apiRsp := response.GetPostRedeployedInwardReportsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostRedeployedInwardReports API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostRedeployedInwardReportsRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

// GetPostRedeployedOutwardReportsHandler to fetch the report on redeployed posts from an office by circle office id.
// @Summary      fetches the report on redeployed posts from an office by circle office id.
// @Description  retrieves the report on redeployed posts from an office by circle office id.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param office-id path string true "Office ID"
// @Success 200 {object} response.GetPostRedeployedOutwardReportsAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/post-redeployed-outward-reports/:office-id [get]
func (pmh *PosttoPostMappingrHandler) GetPostRedeployedOutwardReportsHandler(ctx *gin.Context) {
	var req GetPostRedeployedInwardReportsRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetPostRedeployedOutwardReports(ctx, req.OfficeID)
	if err != nil {
		log.Error(ctx, "Database error while fetching reports", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	if len(postsavbl) == 0 {
		log.Error(ctx, "No data found in database")
		apierrors.HandleDBError(ctx, errors.New("No data found in database"))
		return
	}
	rsp := response.NewGetPostRedeployedOutwardReportsResponse(postsavbl)

	apiRsp := response.GetPostRedeployedOutwardReportsAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetPostRedeployedOutwardReports API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetPostRedeployedOutwardReportsRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}

// SavePostRedeploymentHandler2 to save the post redeployment entries in the database and upload file to minIO.
// @Summary      to save the post redeployment entries in the database and upload file to minIO.
// @Description  to save the post redeployment entries in the database and upload file to minIO.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param request body SavePostRedeploymentRequest true "Save Post Redeployment Request"
// @Success 201 {object} response.SavePostRedeployment2APIResponse "Success - Post redeployed successfully"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/save-post-redeployment2 [post]
func (pmh *PosttoPostMappingrHandler) SavePostRedeploymentHandler2(ctx *gin.Context) {
	jsonData := ctx.PostForm("data")
	if strings.TrimSpace(jsonData) == "" {
		log.Error(ctx, "No 'data' field found in form or it is empty")
		apierrors.HandleValidationError(ctx, errors.New("missing data field"))
		return
	}
	var req SavePostRedeploymentRequest
	if err := json.Unmarshal([]byte(jsonData), &req); err != nil {
		log.Error(ctx, "Error unmarshalling request: %s", err.Error())
		apierrors.HandleMarshalError(ctx, err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		log.Error(ctx, "Error validating request: %s", err.Error())
		apierrors.HandleValidationError(ctx, err)
		return
	}
	valid, err := pmh.svc.IsPostOfficeMappingValid(ctx, req.PostID, req.RedeploymentFromOfficeID)
	if err != nil {
		log.Error(ctx, "Failed to validate post-office mapping: %v", err)
		apierrors.HandleDBError(ctx, err)
		return
	}
	if !valid {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Post ID and Redeployment From Office ID do not match"})
		return
	}
	// // Check office compatibility
	// compatible, err := pmh.svc.CheckOfficeCompatibility(ctx, req.PostID, req.RedeploymentToOfficeID)
	// if err != nil {
	// 	log.Error(ctx, "Redeployment to office is not compatible: %s", err.Error())
	// 	apierrors.HandleDBError(ctx, err)
	// 	return
	// }
	// if !compatible {
	// 	ctx.JSON(400, gin.H{"error": "Target office is not compatible for redeployment"})
	// 	return
	// }

	// File upload handling
	officeFile, header, err := ctx.Request.FormFile("office_file")
	if err != nil {
		log.Error(ctx, "File upload missing or error: %s", err.Error())
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "File upload is required"})
		return
	}
	defer officeFile.Close()

	fileExt := strings.ToLower(strings.TrimPrefix(filepath.Ext(header.Filename), "."))
	allowedFileExtensions := map[string]bool{"pdf": true}

	if header.Size <= 0 {
		log.Error(ctx, "Invalid file: Size is zero")
		apierrors.HandleBindingError(ctx, errors.New("invalid File"))
		return
	}
	if !allowedFileExtensions[fileExt] {
		log.Error(ctx, "Invalid file type: %s", fileExt)
		apierrors.HandleValidationError(ctx, errors.New("invalid File Type"))
		return
	}

	// Add timestamp to filename
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%d_%d_%d_%s.%s", req.PostID, req.RedeploymentToOfficeID, req.RedeploymentFromOfficeID, timestamp, fileExt)
	filename = strings.ReplaceAll(filename, " ", "_")
	filename = strings.ReplaceAll(filename, "-", "_")
	officeFilePath := path.Join("postredeployment/", filename)
	log.Debug(ctx, "File path resolved: %s", officeFilePath)

	// Map request to domain structs
	postRedeploymentLog := ToPostRedeploymentLog(req)
	postManagementMasterUpdate := ToPostManagementMasterUpdate(req)

	// Begin a DB transaction inside the service layer
	tx, err := pmh.svc.BeginTransaction(ctx)
	if err != nil {
		log.Error(ctx, "Failed to begin DB transaction: %v", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Ensure rollback on panic or error
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback(ctx)
			panic(r)
		}
	}()

	// Pass the file path to the repo function (add as a parameter as needed)
	err = pmh.svc.SavePostRedeploymentRepo2(ctx, tx, postRedeploymentLog, postManagementMasterUpdate, filename)
	if err != nil {
		_ = tx.Rollback(ctx)
		log.Error(ctx, "Database error while post redeployment: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	// Commit transaction before file upload
	if err := tx.Commit(ctx); err != nil {
		log.Error(ctx, "Failed to commit transaction: %v", err)
		apierrors.HandleDBError(ctx, err)
		return
	}
	uploadedFile := &domain.MinioFile{
		FilePath:    officeFilePath,
		FileSize:    header.Size,
		ContentType: header.Header.Get("Content-Type"),
		File:        officeFile,
	}

	// Upload the file to MinIO
	if err := pmh.svc.UploadFile(ctx, uploadedFile); err != nil {
		log.Error(ctx, "Failed to upload file: %s", err.Error())
		// Compensate DB changes because file upload failed
		if compErr := pmh.svc.CompensateFailedRedeployment(ctx, req.PostID, officeFilePath); compErr != nil {
			log.Error(ctx, "Failed to compensate redeployment after upload failure: %v", compErr)
		}
		apierrors.HandleDBError(ctx, err)
		return
	}
	apiRsp := response.SavePostRedeployment2APIResponse{
		StatusCodeAndMessage: port.TransferSuccess,
		FileName:             officeFilePath, // Optionally include in response struct
	}

	log.Debug(ctx, "SavePostRedeploymentHandler response: %v", apiRsp)
	handleCreateSuccess(ctx, apiRsp)
}

// SavePostRedeploymentHandler to save the post redeployment entries in the database.
// @Summary      to save the post redeployment entries in the database.
// @Description  to save the post redeployment entries in the database.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param request body SavePostRedeploymentRequest true "Save Post Redeployment Request"
// @Success 201 {object} response.SavePostRedeploymentAPIResponse "Success - Post redeployed successfully"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router       /ptop-mappings/save-post-redployment [post]
func (pmh *PosttoPostMappingrHandler) SavePostRedeploymentHandler(ctx *gin.Context) {

	var req SavePostRedeploymentRequest
	if err := apiutility.BindAndValidate(ctx, &req, false, false, true); err != nil {
		log.Error(ctx, err)
		return
	}
	valid, err := pmh.svc.IsPostOfficeMappingValid(ctx, req.PostID, req.RedeploymentFromOfficeID)
	if err != nil {
		log.Error(ctx, "Failed to validate post-office mapping: %v", err)
		apierrors.HandleDBError(ctx, err)
		return
	}
	if !valid {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Post ID and Redeployment From Office ID do not match"})
		return
	}
	// // Check office compatibility
	// compatible, err := pmh.svc.CheckOfficeCompatibility(ctx, req.PostID, req.RedeploymentToOfficeID)
	// if err != nil {
	// 	log.Error(ctx, "Redeployment to office is not compatible: %s", err.Error())
	// 	apierrors.HandleDBError(ctx, err)
	// 	return
	// }
	// if !compatible {
	// 	ctx.JSON(400, gin.H{"error": "Target office is not compatible for redeployment"})
	// 	return
	// }

	// Map request to domain structs
	postRedeploymentLog := ToPostRedeploymentLog(req)
	postManagementMasterUpdate := ToPostManagementMasterUpdate(req)

	// Begin a DB transaction inside the service layer
	tx, err := pmh.svc.BeginTransaction(ctx)
	if err != nil {
		log.Error(ctx, "Failed to begin DB transaction: %v", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	// Ensure rollback on panic or error
	defer func() {
		if r := recover(); r != nil {
			tx.Rollback(ctx)
			panic(r)
		}
	}()

	err = pmh.svc.SavePostRedeploymentRepo(ctx, tx, postRedeploymentLog, postManagementMasterUpdate)
	if err != nil {
		_ = tx.Rollback(ctx)
		log.Error(ctx, "Database error while post redeployment: %s", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	// Commit transaction before file upload
	if err := tx.Commit(ctx); err != nil {
		log.Error(ctx, "Failed to commit transaction: %v", err)
		apierrors.HandleDBError(ctx, err)
		return
	}

	apiRsp := response.SavePostRedeploymentAPIResponse{
		StatusCodeAndMessage: port.CreateSuccess,
		Data:                 "Post redeployment recorded successfully",
	}

	log.Debug(ctx, "SavePostRedeploymentHandler response:  %s", apiRsp)
	handleCreateSuccess(ctx, apiRsp)
}

// GetAuthorityDetailsForRedeployedPostHandler to check redeployed post holds authority charges.
// @Summary      fetches the authority charges for a redeployed post.
// @Description  retrieves the authority charges for a redeployed post.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param post-id path int true "Post ID"
// @Success 200 {object} response.GetRedeployedPostAuthorityChargesAPIResponse "Success -Successfully retrieved Data"
// @Failure      400   {object}  apierrors.APIErrorResponse            "Bad Request - Invalid request format"
// @Failure      401   {object}  apierrors.APIErrorResponse            "Unauthorized error - Invalid or expired token"
// @Failure      404   {object}  apierrors.APIErrorResponse            "Not Found - No active sessions for the user"
// @Failure      500   {object}  apierrors.APIErrorResponse            "Internal server error - Unexpected issue occurred"
// @Router /ptop-mappings/redeployed-post-authority-details/{post-id} [get]
func (pmh *PosttoPostMappingrHandler) GetRedeployedPostAuthorityChargesHandler(ctx *gin.Context) {
	var req GetRedeployedPostAuthorityChargesRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}
	postsavbl, err := pmh.svc.GetRedeployedPostAuthorityCharges(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "Database error while fetching reports", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}

	rsp := response.NewGetRedeployedPostAuthorityChargesResponse(postsavbl)

	apiRsp := response.GetRedeployedPostAuthorityChargesAPIResponse{
		StatusCode: 200,
		Message:    "Success",
		Data:       rsp,
		TotalCount: len(rsp), // Now counts offices, not posts
	}

	log.Debug(ctx, "GetRedeployedPostAuthorityCharges API Response to be sent: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type GetRedeployedPostAuthorityChargesRequest struct {
	PostID int64 `uri:"post-id" validate:"required"`
}

// UpdateRedeployedPostAuthorityChargesHandler updates the authority details for a redeployed post by setting the post_id to NULL.
// @Summary      Update  authority details for redeployed post
// @Description  Sets post_id to NULL for the given post ID.
// @Tags         Ptop-mappings
// @Accept       json
// @Produce      json
// @Param        post-id path int true "Post ID"
// @Success      200 {object} response.GenericSuccessResponse "Success - Records updated"
// @Failure      400 {object} apierrors.APIErrorResponse "Bad Request - Invalid request format"
// @Failure      401 {object} apierrors.APIErrorResponse "Unauthorized - Invalid or expired token"
// @Failure      404 {object} apierrors.APIErrorResponse "Not Found - No matching records for the post ID"
// @Failure      500 {object} apierrors.APIErrorResponse "Internal Server Error - Unexpected issue occurred"
// @Router       /ptop-mappings/update-redeployed-post-authority-details/{post-id} [put]
func (pmh *PosttoPostMappingrHandler) UpdateRedeployedPostAuthorityChargesHandler(ctx *gin.Context) {
	var req UpdateRedeployedPostAuthorityChargesRequest

	if err := apiutility.BindAndValidate(ctx, &req, true, false, false); err != nil {
		log.Error(ctx, err)
		return
	}

	// Call service to update records
	updatedRecords, err := pmh.svc.UpdateRedeployedPostAuthorityCharges(ctx, req.PostID)
	if err != nil {
		log.Error(ctx, "Database error while updating records", err.Error())
		apierrors.HandleDBError(ctx, err)
		return
	}
	// Handle zero records affected
	if updatedRecords == 0 {
		apiRsp := response.GenericSuccessResponse{
			StatusCode: 200,
			Message:    "No matching records found to update",
		}
		log.Debug(ctx, "UpdateRedeployedPostAuthorityChargesHandler response: ", apiRsp)
		handleSuccess(ctx, apiRsp)
		return
	}
	message := fmt.Sprintf("Successfully updated %d record(s)", updatedRecords)

	// Create a generic response with only a summary
	apiRsp := response.GenericSuccessResponse{
		StatusCode: 200,
		Message:    message,
	}

	log.Debug(ctx, "UpdateMasterAuthoritiesDetailsForRedeployedPostHandler response: ", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type UpdateRedeployedPostAuthorityChargesRequest struct {
	PostID int `uri:"post-id" validate:"required"`
}
