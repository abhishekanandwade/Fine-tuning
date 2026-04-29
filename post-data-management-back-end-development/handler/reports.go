package handler

import (
	"fmt"
	"net/http"
	"pmdm/core/port"
	"pmdm/handler/response"

	"github.com/gin-gonic/gin"
	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
	validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
)

// FetchAllPostsByDivisionHandler godoc
// @Summary       Fetch all posts under a division
// @Description   Fetches all posts for offices belonging to the given division office ID. Only PDN and RDN office types are allowed.
// @Tags          Post Management
// @Accept        json
// @Produce       json
// @Param         office-id  path  int  true  "Office ID (must be PDN or RDN type)"
// @Success       200  {object}  response.FetchPostsByOfficeIDAllAPIResponse2  "Successful response with posts"
// @Failure       400  {object}  apierrors.APIErrorResponse  "Validation or binding error"
// @Failure       500  {object}  apierrors.APIErrorResponse  "Internal server error"
// @Router        /post-management/office-post-details/{office-id}/division-report [get]
func (pmh *PostManagementHandler) FetchAllPostsByDivisionHandler(ctx *gin.Context) {
	var req DivisionEstablishmentReportRequest
	if err := ctx.ShouldBindUri(&req); err != nil {
		apierrors.HandleBindingError(ctx, err)
		log.Error(ctx, "Binding failed for DivisionEstablishmentReportRequest: %s", err)
		return
	}
	if err := validation.ValidateStruct(req); err != nil {
		apierrors.HandleValidationError(ctx, err)
		log.Error(ctx, "Validation failed for DivisionEstablishmentReportRequest: %s", err)
		return
	}

	// Validate that the office is of type PDN or RDN
	officeType, err := pmh.svc.GetOfficeTypeByID(ctx, req.OfficeID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "GetOfficeTypeByID failed for office %d: %s", req.OfficeID, err.Error())
		return
	}
	if officeType != "PDN" && officeType != "RDN" {
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("office %d is of type %s; division report is only available for PDN or RDN offices", req.OfficeID, officeType),
		})
		return
	}

	masterList, err := pmh.svc.FetchAllPostsByDivisionOfficeID(ctx, req.OfficeID)
	if err != nil {
		apierrors.HandleDBError(ctx, err)
		log.Error(ctx, "FetchAllPostsByDivisionOfficeID failed: %s", err.Error())
		return
	}

	rsp := response.NewFetchPostsByOfficeIDAllResponse2(masterList)
	metadata := port.NewMetaDataResponse(0, 0, len(rsp))
	apiRsp := response.FetchPostsByOfficeIDAllAPIResponse2{
		StatusCodeAndMessage: port.ListSuccess,
		MetaDataResponse:     metadata,
		Data:                 rsp,
	}

	log.Debug(ctx, "FetchAllPostsByDivisionHandler response: %v", apiRsp)
	handleSuccess(ctx, apiRsp)
}

type DivisionEstablishmentReportRequest struct {
	OfficeID int `uri:"office-id" validate:"required"`
}
