package response

import (
	"fmt"
	"pmdm/core/domain"
	"pmdm/core/port"

	//"pmdm/core/port"
	"time"

	"github.com/volatiletech/null/v9"
)

type ListCadresResponse struct {
	CadreId   int    `json:"cadre_id"`
	CadreName string `json:"cadre_name"`
	GroupName string `json:"group_name"`
}

func NewListCadresResponse(data []domain.CadreMaster) []ListCadresResponse {
	var response []ListCadresResponse
	for _, cadre := range data {
		payrollResponse := ListCadresResponse{
			CadreId:   int(cadre.CadreID.Int32),
			CadreName: cadre.CadreName.String,
			GroupName: cadre.GroupName.String,
		}
		response = append(response, payrollResponse)
	}
	return response
}

type ListCadresAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListCadresResponse `json:"data"`
}

type ListAllCadresResponse struct {
	CadreID   int32     `json:"cadre_id"`
	CadreName string    `json:"cadre_name"`
	GroupName string    `json:"group_name"`
	PayLevel  int32     `json:"pay_level"`
	GradePay  int32     `json:"grade_pay"`
	ValidFrom time.Time `json:"valid_from"`
	ValidTo   time.Time `json:"valid_to"`
	Status    string    `json:"status"`
	Remarks   string    `json:"remarks"`
	GroupCode int16     `json:"group_id"`
}

func NewListAllCadresResponse(data []domain.CadreMaster) []ListAllCadresResponse {
	var response []ListAllCadresResponse
	for _, cadre := range data {
		allCadreResponse := ListAllCadresResponse{
			CadreID:   cadre.CadreID.Int32,
			CadreName: cadre.CadreName.String,
			GroupName: cadre.GroupName.String,
			PayLevel:  cadre.PayLevel.Int32,
			GradePay:  cadre.GradePay.Int32,
			ValidFrom: cadre.ValidFrom.Time,
			ValidTo:   cadre.ValidTo.Time,
			Status:    cadre.Status.String,
			Remarks:   cadre.Remarks.String,
			GroupCode: cadre.GroupCode.Int16,
		}
		response = append(response, allCadreResponse)
	}
	return response
}

type ListAllCadresAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListAllCadresResponse `json:"data"`
}

type GetPostNameMasterResponse struct {
	PostName string `db:"post_name" json:"post_name"`
	Group    string `db:"group" json:"group"`
	Cadre    string `db:"cadre" json:"cadre"`
}

func NewGetPostNameMasterResponse(data []domain.PostManagementMaster5) []GetPostNameMasterResponse {
	var response []GetPostNameMasterResponse
	for _, postName := range data {
		payrollResponse := GetPostNameMasterResponse{
			PostName: postName.PostName,
			Group:    postName.Group,
			Cadre:    postName.Cadre,
		}
		response = append(response, payrollResponse)
	}
	return response
}

type GetPostNameMasterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []GetPostNameMasterResponse `json:"data"`
}

type PostManagementGroupByCadreCountByOfficeIDResponse struct {
	CadreName    string `json:"cadre_name"`    // Name of the cadre
	FilledStatus string `json:"filled_status"` // Status of whether the post is filled
	Count        int    `json:"count"`         // Number of posts for this combination
}

func NewPostManagementGroupByCadreCountByOfficeIDResponse(data []domain.PostManagementMaster) []PostManagementGroupByCadreCountByOfficeIDResponse {
	var response []PostManagementGroupByCadreCountByOfficeIDResponse
	for _, item := range data {
		resp := PostManagementGroupByCadreCountByOfficeIDResponse{
			CadreName:    item.CadreName.String,
			FilledStatus: item.FilledStatus.String,
			Count:        int(item.Count.Int32),
		}
		response = append(response, resp)
	}
	return response
}

type PostManagementGroupByCadreCountByOfficeIDAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`                                    // Status code and message inline
	port.MetaDataResponse     `json:",inline"`                                    // Metadata response inline
	Data                      []PostManagementGroupByCadreCountByOfficeIDResponse `json:"data"` // Array of post group information
}

type PostManagementByOfficeAndPostResponse struct {
	PostID             int       `json:"post_id"`
	OfficeID           int       `json:"office_id"`
	OfficeName         string    `json:"office_name"`
	PostName           string    `json:"post_name"`
	CadreName          string    `json:"cadre_name"`
	GroupName          string    `json:"group_name"`
	FilledStatus       string    `json:"filled_status"`
	ApprovedBy         string    `json:"approved_by,omitempty"`
	ApprovedOn         time.Time `json:"approved_on,omitempty"`
	AllowancesAttached bool      `json:"allowances_attached"`
	Designation        string    `json:"designation"`
	PayLevel           int       `json:"pay_level"`
	GradePay           int       `json:"grade_pay"`
	Remarks            string    `json:"remarks,omitempty"`
}

func NewPostManagementByOfficeAndPostResponse(data []domain.PostManagementMaster) []PostManagementByOfficeAndPostResponse {
	var response []PostManagementByOfficeAndPostResponse
	for _, item := range data {
		resp := PostManagementByOfficeAndPostResponse{
			PostID:             int(item.PostID.Int32),
			OfficeID:           int(item.OfficeID.Int32),
			OfficeName:         item.OfficeName.String,
			PostName:           item.PostName.String,
			CadreName:          item.CadreName.String,
			GroupName:          item.GroupName.String,
			FilledStatus:       item.FilledStatus.String,
			ApprovedBy:         item.ApprovedBy.String,
			ApprovedOn:         item.ApprovedOn.Time,
			AllowancesAttached: item.AllowancesAttached.Bool,
			Designation:        item.Designation.String,
			PayLevel:           int(item.PayLevel.Int32),
			GradePay:           int(item.GradePay.Int32),
			Remarks:            item.Remarks.String,
		}
		response = append(response, resp)
	}
	return response
}

type PostManagementByOfficeAndPostAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []PostManagementByOfficeAndPostResponse `json:"data"`
}

type GetPostManagementMasterWithMakerResponse struct {
	OfficeID        int       `json:"office_id"`
	OfficeName      string    `json:"office_name"`
	PostID          int       `json:"post_id"`
	PostName        string    `json:"post_name"`
	GroupId         int       `json:"group_id"`
	GroupName       string    `json:"group_name"`
	Designation     string    `json:"designation"`
	FilledStatus    string    `json:"filled_status"`
	PermanentStatus bool      `json:"permanent_status"`
	ApproveStatus   string    `json:"approve_status,omitempty"`
	Status          string    `json:"status"`
	Remarks         string    `json:"remarks,omitempty"`
	NewOfficeID     int       `json:"new_office_id,omitempty"`
	NewOfficeName   string    `json:"new_office_name,omitempty"`
	ExchangePostID  int       `json:"exchange_post_id,omitempty"`
	OrderDate       time.Time `json:"order_date,omitempty"`
	CadreID         int       `json:"cadre_id"`
	CadreName       string    `json:"cadre_name"`
	EmployeeGroup   string    `json:"employee_group"`
}

func NewGetPostManagementMasterWithMakerResponse(data []domain.PostManagementMaker) []GetPostManagementMasterWithMakerResponse {
	var response []GetPostManagementMasterWithMakerResponse
	for _, item := range data {
		resp := GetPostManagementMasterWithMakerResponse{
			OfficeID:        item.OfficeID,
			OfficeName:      item.OfficeName,
			PostID:          item.PostID,
			PostName:        item.PostName,
			GroupId:         item.GroupId,
			GroupName:       item.GroupName,
			Designation:     item.Designation,
			FilledStatus:    item.FilledStatus,
			PermanentStatus: item.PermanentStatus,
			ApproveStatus:   item.PostStatus, // Assuming ApproveStatus corresponds to PostStatus
			Status:          item.Status,
			Remarks:         item.Remarks,
			NewOfficeID:     item.EstablishmentRegisterID,   // Assuming NewOfficeID maps to EstablishmentRegisterID
			NewOfficeName:   item.EstablishmentRegisterName, // Assuming NewOfficeName maps to EstablishmentRegisterName
			ExchangePostID:  item.DesignationId,             // Assuming ExchangePostID maps to DesignationId
			OrderDate:       item.OrderDate,
			CadreID:         item.CadreID,
			CadreName:       item.CadreName,
			EmployeeGroup:   item.EmployeeGroup,
		}
		response = append(response, resp)
	}
	return response
}

type GetPostManagementMasterWithMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []GetPostManagementMasterWithMakerResponse `json:"data"`
}

type FetchSurplusPostRecordByApproverPostIDResponse struct {
	PostID          int       `json:"post_id"`
	PostName        string    `json:"post_name"`
	OfficeID        int       `json:"office_id"`
	OfficeName      string    `json:"office_name"`
	NewOfficeID     int       `json:"new_office_id,omitempty"`
	NewOfficeName   string    `json:"new_office_name,omitempty"`
	CadreID         int       `json:"cadre_id"`
	CadreName       string    `json:"cadre_name"`
	GroupID         int       `json:"group_id"`
	GroupName       string    `json:"group_name"`
	EmployeeGroup   string    `json:"employee_group"`
	DesignationID   int       `json:"designation_id"`
	Designation     string    `json:"designation"`
	PayLevel        int       `json:"pay_level"`
	GradePay        int       `json:"grade_pay"`
	PermanentStatus bool      `json:"permanent_status"`
	PostStatus      string    `json:"post_status"`
	FilledStatus    string    `json:"filled_status"`
	ExchangePostID  int       `json:"exchange_post_id,omitempty"`
	ApproveStatus   string    `json:"approve_status,omitempty"`
	Status          string    `json:"status"`
	OrderDate       time.Time `json:"order_date,omitempty"`
	Remarks         string    `json:"remarks,omitempty"`
}

// NewFetchSurplusPostRecordByApproverPostIDResponse converts service data into the API response format.
func NewFetchSurplusPostRecordByApproverPostIDResponse(data []domain.PostManagementMaker) []FetchSurplusPostRecordByApproverPostIDResponse {
	var response []FetchSurplusPostRecordByApproverPostIDResponse
	for _, item := range data {
		resp := FetchSurplusPostRecordByApproverPostIDResponse{
			PostID:          item.PostID,
			PostName:        item.PostName,
			OfficeID:        item.OfficeID,
			OfficeName:      item.OfficeName,
			NewOfficeID:     item.NewOfficeID,
			NewOfficeName:   item.NewOfficeName,
			CadreID:         item.CadreID,
			CadreName:       item.CadreName,
			GroupID:         item.GroupId, // Corrected GroupID mapping
			GroupName:       item.GroupName,
			EmployeeGroup:   item.EmployeeGroup,
			DesignationID:   item.DesignationId,
			Designation:     item.Designation,
			PayLevel:        item.PayLevel,
			GradePay:        item.GradePay,
			PermanentStatus: item.PermanentStatus,
			PostStatus:      item.PostStatus,
			FilledStatus:    item.FilledStatus,
			ExchangePostID:  item.ExchangePostID,
			ApproveStatus:   item.ApproveStatus,
			Status:          item.Status,
			OrderDate:       item.OrderDate,
			Remarks:         item.Remarks,
		}
		response = append(response, resp)
	}
	return response
}

type FetchSurplusPostRecordByApproverPostIDAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []FetchSurplusPostRecordByApproverPostIDResponse `json:"data"`
}

type PostManagementByOfficeIDAndStatusResponse struct {
	OfficeID      int    `json:"office_id"`
	PostID        int    `json:"post_id"`
	PostName      string `json:"post_name"`
	CadreName     string `json:"cadre_name"`
	Designation   string `json:"designation"`
	FilledStatus  string `json:"filled_status"`
	Status        string `json:"status"`
	GroupId       int    `json:"group_id"` // Use GroupId instead of GroupID
	EmployeeGroup string `json:"employee_group,omitempty"`
	CadreID       int    `json:"cadre_id"`
}

func NewPostManagementByOfficeIDAndStatusResponse(data []domain.PostManagementMaster) []PostManagementByOfficeIDAndStatusResponse {
	var response []PostManagementByOfficeIDAndStatusResponse
	for _, item := range data {
		resp := PostManagementByOfficeIDAndStatusResponse{
			OfficeID:      int(item.OfficeID.Int32),
			PostID:        int(item.PostID.Int32),
			PostName:      item.PostName.String,
			CadreName:     item.CadreName.String,
			Designation:   item.Designation.String,
			FilledStatus:  item.FilledStatus.String,
			Status:        item.Status.String,
			GroupId:       int(item.GroupId.Int32), // Correct field name assumed here
			EmployeeGroup: item.EmployeeGroup.String,
			CadreID:       int(item.CadreID.Int32),
		}
		response = append(response, resp)
	}
	return response
}

type PostManagementByOfficeIDAndStatusAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []PostManagementByOfficeIDAndStatusResponse `json:"data"`
}

type FetchPostsByOfficeIDResponse struct {
	OfficeID                  int       `json:"office_id"`
	PostID                    int       `json:"post_id"`
	PostName                  string    `json:"post_name"`
	OfficeName                string    `json:"office_name"`
	GroupId                   int       `json:"group_id"`
	GroupName                 string    `json:"group_name,omitempty"`
	CadreID                   int       `json:"cadre_id"`
	CadreName                 string    `json:"cadre_name"`
	DesignationID             int       `json:"designation_id"`
	Designation               string    `json:"designation"`
	FilledStatus              string    `json:"filled_status"`
	PostStatus                string    `json:"post_status"`
	Status                    string    `json:"status"`
	PermanentStatus           bool      `json:"permanent_status"`
	AllowancesAttached        bool      `json:"allowances_attached"`
	AllowanceDescription      string    `json:"allowance_description,omitempty"`
	CreatedBy                 string    `json:"created_by"`
	CreatedOn                 time.Time `json:"created_on"`
	ApprovedBy                string    `json:"approved_by,omitempty"`
	ApprovedOn                time.Time `json:"approved_on,omitempty"`
	UpdatedBy                 string    `json:"updated_by,omitempty"`
	UpdatedOn                 time.Time `json:"updated_on,omitempty"`
	ValidFrom                 time.Time `json:"valid_from,omitempty"`
	ValidTo                   time.Time `json:"valid_to,omitempty"`
	OrderCaseMark             string    `json:"order_case_mark,omitempty"`
	OrderDate                 time.Time `json:"order_date,omitempty"`
	UploadOrderDocName        string    `json:"upload_order_doc_name,omitempty"`
	EstablishmentRegisterID   int       `json:"establishment_register_id,omitempty"`
	EstablishmentRegisterName string    `json:"establishment_register_name,omitempty"`
	EmployeeGroup             string    `json:"employee_group,omitempty"`
	SanctionedStrength        int       `json:"sanctioned_strength,omitempty"`
	ExchangePostID            int       `json:"exchange_post_id,omitempty"`
	Remarks                   string    `json:"remarks,omitempty"`
	ApproveStatus             string    `json:"approve_status,omitempty"`
	ApprovePostID             string    `json:"approve_post_id,omitempty"`
	EmployeeID                int32     `json:"employee_id" db:"employee_id"`     // Employee ID from kafka_employee_master
	EmployeeName              string    `json:"employee_name" db:"employee_name"` // Employee full name (concatenated)
}

func NewFetchPostsByOfficeIDResponse(
	masters []domain.PostManagementMaster,
	makers []domain.PostManagementMaker,
) []FetchPostsByOfficeIDResponse {
	var response []FetchPostsByOfficeIDResponse

	for _, item := range masters {
		resp := FetchPostsByOfficeIDResponse{
			OfficeID:                  int(item.OfficeID.Int32),
			PostID:                    int(item.PostID.Int32),
			PostName:                  item.PostName.String,
			OfficeName:                item.OfficeName.String,
			GroupId:                   int(item.GroupId.Int32),
			GroupName:                 item.GroupName.String,
			CadreID:                   int(item.CadreID.Int32),
			CadreName:                 item.CadreName.String,
			DesignationID:             int(item.DesignationId.Int32),
			Designation:               item.Designation.String,
			FilledStatus:              item.FilledStatus.String,
			PostStatus:                item.PostStatus.String,
			Status:                    item.Status.String,
			PermanentStatus:           item.PermanentStatus.Bool,
			AllowancesAttached:        item.AllowancesAttached.Bool,
			AllowanceDescription:      item.AllowanceDescription.String,
			CreatedBy:                 item.CreatedBy.String,
			CreatedOn:                 item.CreatedOn.Time,
			ApprovedBy:                item.ApprovedBy.String,
			ApprovedOn:                item.ApprovedOn.Time,
			UpdatedBy:                 item.UpdatedBy.String,
			UpdatedOn:                 item.UpdatedOn.Time,
			ValidFrom:                 item.ValidFrom.Time,
			ValidTo:                   item.ValidTo.Time,
			OrderCaseMark:             item.OrderCaseMark.String,
			OrderDate:                 item.OrderDate.Time,
			UploadOrderDocName:        item.UploadOrderDocName.String,
			EstablishmentRegisterID:   int(item.EstablishmentRegisterID.Int32),
			EstablishmentRegisterName: item.EstablishmentRegisterName.String,
			EmployeeGroup:             item.EmployeeGroup.String,
			SanctionedStrength:        int(item.SanctionedStrength.Int32),
			Remarks:                   item.Remarks.String,
			ApproveStatus:             item.ApproveStatus.String,
			EmployeeID:                item.EmployeeID.Int32,
			EmployeeName:              item.EmployeeName.String,
		}
		response = append(response, resp)
	}

	for _, item := range makers {
		resp := FetchPostsByOfficeIDResponse{
			OfficeID:                  item.OfficeID,
			PostID:                    item.PostID,
			PostName:                  item.PostName,
			OfficeName:                item.OfficeName,
			GroupId:                   item.GroupId,
			CadreID:                   item.CadreID,
			CadreName:                 item.CadreName,
			DesignationID:             item.DesignationId,
			Designation:               item.Designation,
			FilledStatus:              item.FilledStatus,
			PostStatus:                item.PostStatus,
			Status:                    item.Status,
			PermanentStatus:           item.PermanentStatus,
			AllowancesAttached:        item.AllowancesAttached,
			AllowanceDescription:      item.AllowanceDescription,
			CreatedBy:                 item.CreatedBy,
			CreatedOn:                 item.CreatedDate,
			ValidFrom:                 item.ValidFrom,
			ValidTo:                   item.ValidTo,
			OrderCaseMark:             item.OrderCaseMark,
			OrderDate:                 item.OrderDate,
			UploadOrderDocName:        item.UploadOrderDocName,
			EstablishmentRegisterID:   item.EstablishmentRegisterID,
			EstablishmentRegisterName: item.EstablishmentRegisterName,
			EmployeeGroup:             item.EmployeeGroup,
			SanctionedStrength:        item.SanctionedStrength,
			ExchangePostID:            item.ExchangePostID,
			Remarks:                   item.Remarks,
			ApproveStatus:             item.ApproveStatus,
			ApprovePostID:             item.ApprovePostID,
		}
		response = append(response, resp)
	}

	return response
}

type FetchPostsByOfficeIDAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []FetchPostsByOfficeIDResponse `json:"data"`
}
type FetchPostsByOfficeIDAPIResponse2 struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []FetchPostsByOfficeIDResponseNew `json:"data"`
}

type PostManagementWithPendingStatusOfMakerResponse struct {
	OfficeID        int       `json:"office_id"`
	PostName        string    `json:"post_name"`
	OfficeName      string    `json:"office_name"`
	GroupID         int       `json:"group_id"`
	FilledStatus    string    `json:"filled_status"`
	PostID          int       `json:"post_id"`
	Designation     string    `json:"designation"`
	PermanentStatus bool      `json:"permanent_status"`
	ApproveStatus   string    `json:"approve_status"`
	Status          string    `json:"status"`
	NewOfficeID     int       `json:"new_office_id"`
	NewOfficeName   string    `json:"new_office_name"`
	Remarks         string    `json:"remarks"`
	ExchangePostID  int       `json:"exchange_post_id"`
	OrderDate       time.Time `json:"order_date"`
	EmployeeGroup   string    `json:"employee_group"`
	CadreID         int       `json:"cadre_id"`
	CadreName       string    `json:"cadre_name"`
	PayLevel        int       `json:"pay_level"`

	// Old record fields from the main table
	OldCadreID       int    `json:"old_cadre_id"`
	OldCadreName     string `json:"old_cadre_name"`
	OldGroupID       int    `json:"old_group_id"`
	OldDesignation   string `json:"old_designation"`
	OldPayLevel      string `json:"old_pay_level"`
	OldGradePay      string `json:"old_grade_pay"`
	OldStatus        string `json:"old_status"`
	OldEmployeeGroup string `json:"old_employee_group"`
	OldPostName      string `json:"old_post_name"`
}

func NewPostManagementWithPendingStatusOfMakerResponse(data []domain.PostManagementMaker1) []PostManagementWithPendingStatusOfMakerResponse {
	var response []PostManagementWithPendingStatusOfMakerResponse
	for _, item := range data {
		resp := PostManagementWithPendingStatusOfMakerResponse{
			OfficeID:        item.OfficeID,
			PostName:        item.PostName,
			OfficeName:      item.OfficeName,
			GroupID:         item.GroupID,
			FilledStatus:    item.FilledStatus,
			PostID:          item.PostID,
			Designation:     item.Designation,
			PermanentStatus: item.PermanentStatus,
			ApproveStatus:   item.ApproveStatus,
			Status:          item.Status,
			NewOfficeID:     item.NewOfficeID,
			NewOfficeName:   item.NewOfficeName,
			Remarks:         item.Remarks,
			ExchangePostID:  item.ExchangePostID,
			OrderDate:       item.OrderDate,
			EmployeeGroup:   item.EmployeeGroup,
			CadreID:         item.CadreID,
			CadreName:       item.CadreName,
			PayLevel:        item.PayLevel,

			// Old record fields from the main table
			OldCadreID:       item.OldCadreID,
			OldCadreName:     item.OldCadreName,
			OldGroupID:       item.OldGroupID,
			OldDesignation:   item.OldDesignation,
			OldPayLevel:      item.OldPayLevel,
			OldGradePay:      item.OldGradePay,
			OldStatus:        item.OldStatus,
			OldEmployeeGroup: item.OldEmployeeGroup,
			OldPostName:      item.OldPostName,
		}
		response = append(response, resp)
	}
	return response
}

type PostManagementWithPendingStatusOfMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []PostManagementWithPendingStatusOfMakerResponse `json:"data"`
}

type FetchPostsByOfficeIDAndMakerResponse struct {
	PostManagementMakerID     int       `json:"post_management_maker_id"`
	OfficeID                  int       `json:"office_id"`
	PostID                    int       `json:"post_id"`
	PostName                  string    `json:"post_name"`
	OfficeName                string    `json:"office_name"`
	NewOfficeID               int       `json:"new_office_id,omitempty"`
	NewOfficeName             string    `json:"new_office_name,omitempty"`
	GroupId                   int       `json:"group_id"`
	GroupName                 string    `json:"group_name,omitempty"`
	CadreID                   int       `json:"cadre_id"`
	CadreName                 string    `json:"cadre_name"`
	DesignationID             int       `json:"designation_id"`
	Designation               string    `json:"designation"`
	FilledStatus              string    `json:"filled_status"`
	PostStatus                string    `json:"post_status"`
	Status                    string    `json:"status"`
	PermanentStatus           bool      `json:"permanent_status"`
	AllowancesAttached        bool      `json:"allowances_attached"`
	AllowanceDescription      string    `json:"allowance_description,omitempty"`
	CreatedBy                 string    `json:"created_by"`
	CreatedDate               time.Time `json:"created_date"`
	ValidFrom                 time.Time `json:"valid_from,omitempty"`
	ValidTo                   time.Time `json:"valid_to,omitempty"`
	OrderCaseMark             string    `json:"order_case_mark,omitempty"`
	OrderDate                 time.Time `json:"order_date,omitempty"`
	UploadOrderDocName        string    `json:"upload_order_doc_name,omitempty"`
	EstablishmentRegisterID   int       `json:"establishment_register_id,omitempty"`
	EstablishmentRegisterName string    `json:"establishment_register_name,omitempty"`
	EmployeeGroup             string    `json:"employee_group,omitempty"`
	SanctionedStrength        int       `json:"sanctioned_strength,omitempty"`
	Remarks                   string    `json:"remarks,omitempty"`
	ApproveStatus             string    `json:"approve_status"`
	ApprovePostID             string    `json:"approve_post_id"`
	ExchangePostID            int       `json:"exchange_post_id,omitempty"`
}

func NewFetchPostsByOfficeIDAndMakerResponse(data []domain.PostManagementMaker) []FetchPostsByOfficeIDAndMakerResponse {
	var response []FetchPostsByOfficeIDAndMakerResponse
	for _, item := range data {
		resp := FetchPostsByOfficeIDAndMakerResponse{
			PostManagementMakerID:     item.PostManagementMakerID,
			OfficeID:                  item.OfficeID,
			PostID:                    item.PostID,
			PostName:                  item.PostName,
			OfficeName:                item.OfficeName,
			NewOfficeID:               item.NewOfficeID,
			NewOfficeName:             item.NewOfficeName,
			GroupId:                   item.GroupId,
			GroupName:                 item.GroupName,
			CadreID:                   item.CadreID,
			CadreName:                 item.CadreName,
			DesignationID:             item.DesignationId,
			Designation:               item.Designation,
			FilledStatus:              item.FilledStatus,
			PostStatus:                item.PostStatus,
			Status:                    item.Status,
			PermanentStatus:           item.PermanentStatus,
			AllowancesAttached:        item.AllowancesAttached,
			AllowanceDescription:      item.AllowanceDescription,
			CreatedBy:                 item.CreatedBy,
			CreatedDate:               item.CreatedDate,
			ValidFrom:                 item.ValidFrom,
			ValidTo:                   item.ValidTo,
			OrderCaseMark:             item.OrderCaseMark,
			OrderDate:                 item.OrderDate,
			UploadOrderDocName:        item.UploadOrderDocName,
			EstablishmentRegisterID:   item.EstablishmentRegisterID,
			EstablishmentRegisterName: item.EstablishmentRegisterName,
			EmployeeGroup:             item.EmployeeGroup,
			SanctionedStrength:        item.SanctionedStrength,
			Remarks:                   item.Remarks,
			ApproveStatus:             item.ApproveStatus,
			ApprovePostID:             item.ApprovePostID,
			ExchangePostID:            item.ExchangePostID,
		}
		response = append(response, resp)
	}
	return response
}

type FetchPostsByOfficeIDAndMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []FetchPostsByOfficeIDAndMakerResponse `json:"data"`
}

type FetchEstablishmentRegisterResponse struct {
	OfficeID                int    `json:"office_id"`
	PostID                  int    `json:"post_id"`
	PostName                string `json:"post_name"`
	OfficeName              string `json:"office_name"`
	GroupId                 int    `json:"group_id"`
	GroupName               string `json:"group_name"`
	Designation             string `json:"designation"`
	PermanentStatus         bool   `json:"permanent_status"`
	Status                  string `json:"status"`
	CadreID                 int    `json:"cadre_id"`
	CadreName               string `json:"cadre_name"`
	EmployeeGroup           string `json:"employee_group"`
	PayLevel                int    `json:"pay_level"`
	GradePay                int    `json:"grade_pay"`
	Remarks                 string `json:"remarks"`
	EstablishmentRegisterID int    `json:"establishment_register_id"`
}

func NewFetchEstablishmentRegisterResponse(data []domain.PostManagementMaster) []FetchEstablishmentRegisterResponse {
	var response []FetchEstablishmentRegisterResponse
	for _, item := range data {
		resp := FetchEstablishmentRegisterResponse{
			OfficeID:                int(item.OfficeID.Int32),
			PostID:                  int(item.PostID.Int32),
			PostName:                item.PostName.String,
			OfficeName:              item.OfficeName.String,
			GroupId:                 int(item.GroupId.Int32),
			GroupName:               item.GroupName.String,
			Designation:             item.Designation.String,
			PermanentStatus:         item.PermanentStatus.Bool,
			Status:                  item.Status.String,
			CadreID:                 int(item.CadreID.Int32),
			CadreName:               item.CadreName.String,
			EmployeeGroup:           item.EmployeeGroup.String,
			PayLevel:                int(item.PayLevel.Int32),
			GradePay:                int(item.GradePay.Int32),
			Remarks:                 item.Remarks.String,
			EstablishmentRegisterID: int(item.EstablishmentRegisterID.Int32),
		}
		response = append(response, resp)
	}
	return response
}

type FetchEstablishmentRegisterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []FetchEstablishmentRegisterResponse `json:"data"`
}

type EstblishnentRegisterByOfficeResponse struct {
	OfficeID                  int    `json:"office_id"`
	OfficeName                string `json:"office_name"`
	EstablishmentRegisterID   int    `json:"establishment_register_id"`
	EstablishmentRegisterName string `json:"establishment_register_name"`
	SanctionedStrength        int    `json:"sanctioned_strength"`
}

func NewEstblishnentRegisterByOfficeResponse(data domain.PostManagementMaster) EstblishnentRegisterByOfficeResponse {
	return EstblishnentRegisterByOfficeResponse{
		OfficeID:                  int(data.OfficeID.Int32),
		OfficeName:                data.OfficeName.String,
		EstablishmentRegisterID:   int(data.EstablishmentRegisterID.Int32),
		EstablishmentRegisterName: data.EstablishmentRegisterName.String,
		SanctionedStrength:        int(data.SanctionedStrength.Int32),
	}
}

type EstblishnentRegisterByOfficeAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []EstblishnentRegisterByOfficeResponse `json:"data"`
}

type ListAndFilterDesignationsResponse struct {
	DesignationID   int    `json:"designation_id"`
	DesignationName string `json:"designation_name"`
	Designation     string `json:"designation"`
	GroupName       string `json:"group_name"`
	CadreName       string `json:"cadre_name"`
	CadreId         int    `json:"cadre_id"`
	GroupId         int16  `json:"group_id"`
}

func NewListAndFilterDesignationsResponse(
	designationList []domain.DesignationMaster,
	allDesignationTypes []domain.DesignationMaster,
) []ListAndFilterDesignationsResponse {
	var response []ListAndFilterDesignationsResponse

	// Merge the data from both sources
	for _, details := range designationList {
		for _, designationType := range allDesignationTypes {
			if details.DesignationID == designationType.DesignationID {
				combinedResp := ListAndFilterDesignationsResponse{
					DesignationID:   designationType.DesignationID,
					DesignationName: designationType.Designation,
					Designation:     details.Designation,
					GroupName:       details.GroupName,
					CadreName:       details.CadreName,
					CadreId:         details.CadreId,
					GroupId:         details.GroupId,
				}
				response = append(response, combinedResp)
				break
			}
		}
	}

	return response
}

type ListAndFilterDesignationsAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListAndFilterDesignationsResponse `json:"data"`
}

type GetPostMappingMasterResponse struct {
	PostMapID              string `json:"post_map_id"`
	PostMappingColumnName  string `json:"post_mapping_column_name"`
	PostMappingDescription string `json:"post_mapping_description"`
}

func NewGetPostMappingMasterResponse(data []domain.PostMapMaster) []GetPostMappingMasterResponse {
	var response []GetPostMappingMasterResponse
	for _, details := range data {
		resp := GetPostMappingMasterResponse{
			PostMapID:              details.PostMapID,
			PostMappingColumnName:  details.PostMappingColumnName,
			PostMappingDescription: details.PostMappingDescription,
		}
		response = append(response, resp)
	}
	return response
}

type GetPostMappingMasterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []GetPostMappingMasterResponse `json:"data"`
}

type GetAuthorityDetailsByPostIDResponse struct {
	AuthorityName        string `json:"authority_name"`
	AuthorityDescription string `json:"authority_description"`
	AuthorityPost        int    `json:"authority_post"`
	DesignationName      string `json:"designation_name"`
	OfficeID             int    `json:"office_id"`
	OfficeName           string `json:"office_name"`
}

func NewGetAuthorityDetailsByPostIDResponse(
	data map[string]domain.AuthorityDetails,
) []GetAuthorityDetailsByPostIDResponse {
	var response []GetAuthorityDetailsByPostIDResponse
	for _, details := range data {
		resp := GetAuthorityDetailsByPostIDResponse{
			AuthorityName:        details.AuthorityName,
			AuthorityDescription: details.AuthorityDescription,
			AuthorityPost:        details.AuthorityPost,
			DesignationName:      details.DesignationName,
			OfficeID:             details.OfficeID,
			OfficeName:           details.OfficeName,
		}
		response = append(response, resp)
	}
	return response
}

type GetAuthorityDetailsByPostIDAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []GetAuthorityDetailsByPostIDResponse `json:"data"`
}

type GetMasterAuthoritiesDeatilsResponse struct {
	CadreName            string `json:"cadre_name"`
	Designation          string `json:"designation"`
	RoleMappingID        string `json:"role_mapping_id"`
	AuthorityDescription string `json:"authority_description"`
}

func NewGetMasterAuthoritiesDeatilsResponse(
	data []domain.MasterAuthority,
) []GetMasterAuthoritiesDeatilsResponse {
	var response []GetMasterAuthoritiesDeatilsResponse
	for _, authority := range data {
		resp := GetMasterAuthoritiesDeatilsResponse{
			CadreName:            authority.CadreName,
			Designation:          authority.Designation,
			RoleMappingID:        authority.RoleMappingID,
			AuthorityDescription: authority.AuthorityDescription,
		}
		response = append(response, resp)
	}
	return response
}

type GetMasterAuthoritiesDeatilsAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []GetMasterAuthoritiesDeatilsResponse `json:"data"`
}

type GetPostMappingMasterMakerResponse struct {
	AuthorityName        string `json:"authority_name"`
	AuthorityDescription string `json:"authority_description"`
	AuthorityPost        int    `json:"authority_post"`
	DesignationName      string `json:"designation_name"`
	OfficeID             int    `json:"office_id"`
	OfficeName           string `json:"office_name"`
}

func NewGetPostMappingMasterMakerResponse(
	data map[string][]domain.AuthorityDetails,
) []GetPostMappingMasterMakerResponse {
	var response []GetPostMappingMasterMakerResponse
	for _, authorityList := range data {
		for _, details := range authorityList {
			resp := GetPostMappingMasterMakerResponse{
				AuthorityName:        details.AuthorityName,
				AuthorityDescription: details.AuthorityDescription,
				AuthorityPost:        details.AuthorityPost,
				DesignationName:      details.DesignationName,
				OfficeID:             details.OfficeID,
				OfficeName:           details.OfficeName,
			}
			response = append(response, resp)
		}
	}
	return response
}

type GetPostMappingMasterMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []GetPostMappingMasterMakerResponse `json:"data"`
}

type UpdatePostManagementMasterResponse struct {
	OfficeID int64  `json:"office_id"`
	PostID   int64  `json:"post_id"`
	Status   string `json:"status"`
}

func NewUpdatePostManagementMasterResponse(data *domain.PostManagementMaster4) UpdatePostManagementMasterResponse {
	return UpdatePostManagementMasterResponse{
		OfficeID: data.OfficeID,
		PostID:   data.PostID,
		Status:   data.Status,
	}
}

type UpdatePostManagementResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      string `json:"data"`
}

type UpdatePostManagementMasterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      UpdatePostManagementMasterResponse `json:"data"`
}

type ApprovePostManagementMasterResponse struct {
	OfficeID int64  `json:"office_id"`
	PostID   int64  `json:"post_id"`
	Status   string `json:"status"`
}

func NewApprovePostManagementMasterResponse(data *domain.PostManagementMaster4) ApprovePostManagementMasterResponse {
	return ApprovePostManagementMasterResponse{
		OfficeID: data.OfficeID,
		PostID:   data.PostID,
		Status:   data.Status,
	}
}

type ApprovePostManagementMasterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      ApprovePostManagementMasterResponse `json:"data"`
}

type PostManagementChangFilledStatusResponse struct {
	Message string `json:"message"`
}

func NewPostManagementChangFilledStatusResponse(message string) PostManagementChangFilledStatusResponse {
	return PostManagementChangFilledStatusResponse{
		Message: message,
	}
}

type PostManagementChangFilledStatusAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      PostManagementChangFilledStatusResponse `json:"data"`
}

type CreatePostManagementMasterResponse struct {
	PostManagementID int64  `json:"postmanagement_id"`
	OfficeID         int64  `json:"office_id"`
	PostID           int64  `json:"post_id"`
	PostName         string `json:"post_name"`
	OfficeName       string `json:"office_name"`
	Status           string `json:"status"`
}

func NewCreatePostManagementMasterResponse(data []domain.PostManagementMaster2) []CreatePostManagementMasterResponse {
	var response []CreatePostManagementMasterResponse
	for _, item := range data {
		resp := CreatePostManagementMasterResponse{
			PostManagementID: item.PostManagementID,
			OfficeID:         item.OfficeID,
			PostID:           item.PostID,
			PostName:         item.PostName,
			OfficeName:       item.OfficeName,
			Status:           item.Status,
		}
		response = append(response, resp)
	}
	return response
}

type CreatePostManagementMasterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []CreatePostManagementMasterResponse `json:"data"`
}

type RestoredSurplusPostResponse struct {
	Message string `json:"message"`
}

func NewRestoredSurplusPostResponse(message string) RestoredSurplusPostResponse {
	return RestoredSurplusPostResponse{
		Message: message,
	}
}

type RestoredSurplusPostAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      RestoredSurplusPostResponse `json:"data"`
}

type CreatePostManagementMakerResponse struct {
	PostManagementMakerID int    `json:"post_management_maker_id"`
	OfficeID              int64  `json:"office_id"`
	PostID                int64  `json:"post_id"`
	PostName              string `json:"post_name"`
	OfficeName            string `json:"office_name"`
	Status                string `json:"status"`
}

func NewCreatePostManagementMakerResponse(data []*domain.PostManagementMaster3) []CreatePostManagementMakerResponse {
	var response []CreatePostManagementMakerResponse
	for _, item := range data {
		resp := CreatePostManagementMakerResponse{
			PostManagementMakerID: item.PostManagementMakerID,
			OfficeID:              item.OfficeID,
			PostID:                item.PostID,
			PostName:              item.PostName,
			OfficeName:            item.OfficeName,
			Status:                item.Status,
		}
		response = append(response, resp)
	}
	return response
}

type CreatePostManagementMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []CreatePostManagementMakerResponse `json:"data"`
}

type ApprovePostManagementMakerResponse struct {
	PostID int64 `json:"post_id"`
	//OfficeID     int64     `json:"office_id"`
	Status       string    `json:"status"`
	ApprovedBy   string    `json:"approved_by"`
	Remarks      string    `json:"remarks"`
	ApprovedDate time.Time `json:"approved_date"`
}

func NewApprovePostManagementMakerResponse(postIDs []int, approvedBy string) []ApprovePostManagementMakerResponse {
	var response []ApprovePostManagementMakerResponse
	for _, postID := range postIDs {
		resp := ApprovePostManagementMakerResponse{
			PostID: int64(postID),
			//OfficeID:     officeID,
			Status:       "Approved",
			ApprovedBy:   approvedBy,
			Remarks:      fmt.Sprintf("Post %d approved successfully", postID),
			ApprovedDate: time.Now(),
		}
		response = append(response, resp)
	}
	return response
}

type ApprovePostManagementMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []ApprovePostManagementMakerResponse `json:"data"`
}

type ApprovePostManagementMakerForAbolishPostResponse struct {
	PostID        int64     `json:"post_id"`
	OfficeID      int64     `json:"office_id"`
	NewOfficeID   int64     `json:"new_office_id"`
	NewOfficeName string    `json:"new_office_name"`
	Status        string    `json:"status"`
	ApprovedBy    string    `json:"approved_by"`
	ApprovedDate  time.Time `json:"approved_date"`
	Remarks       string    `json:"remarks"`
}

func NewApprovePostManagementMakerForAbolishPostResponse(
	postIDs []int, approvedBy string, officeDetails map[int]struct {
		NewOfficeID   int
		NewOfficeName string
		OfficeID      int
		Status        string
	}) []ApprovePostManagementMakerForAbolishPostResponse {

	var response []ApprovePostManagementMakerForAbolishPostResponse
	for _, postID := range postIDs {
		details := officeDetails[postID]
		resp := ApprovePostManagementMakerForAbolishPostResponse{
			PostID:        int64(postID),
			OfficeID:      int64(details.OfficeID),
			NewOfficeID:   int64(details.NewOfficeID),
			NewOfficeName: details.NewOfficeName,
			Status:        "Approved",
			ApprovedBy:    approvedBy,
			ApprovedDate:  time.Now(),
			Remarks:       fmt.Sprintf("Status: %s, Office ID: %d, New Office ID: %d", details.Status, details.OfficeID, details.NewOfficeID),
		}
		response = append(response, resp)
	}
	return response
}

type ApprovePostManagementMakerForAbolishPostAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []ApprovePostManagementMakerForAbolishPostResponse `json:"data"`
}

type RejectPostManagementMakerResponse struct {
	PostID       int       `json:"post_id"`
	Remarks      string    `json:"remarks"`
	RejectedBy   string    `json:"rejected_by"`
	RejectedDate time.Time `json:"rejected_date"`
}

func NewRejectPostManagementMakerResponse(
	postIDs []int, rejectedBy string,
) []RejectPostManagementMakerResponse {
	var response []RejectPostManagementMakerResponse
	for _, postID := range postIDs {
		resp := RejectPostManagementMakerResponse{
			PostID:       postID,
			Remarks:      fmt.Sprintf("Rejected by %s on %s", rejectedBy, time.Now().Format(time.RFC3339)),
			RejectedBy:   rejectedBy,
			RejectedDate: time.Now(),
		}
		response = append(response, resp)
	}
	return response
}

type RejectPostManagementMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []RejectPostManagementMakerResponse `json:"data"`
}

// Response struct for establishment register creation
type CreateEstablishmentRegisterResponse struct {
	EstablishmentRegisterID   int       `json:"establishment_register_id"`
	OfficeID                  int       `json:"office_id"`
	OfficeName                string    `json:"office_name"`
	EstablishmentRegisterName string    `json:"establishment_register_name"`
	CreatedBy                 string    `json:"created_by"`
	CreatedOn                 time.Time `json:"created_date"`
	Status                    string    `json:"status"`
}

// Helper function to create the response struct
func NewCreateEstablishmentRegisterResponse(est *domain.PostManagementMaster1) CreateEstablishmentRegisterResponse {
	return CreateEstablishmentRegisterResponse{
		EstablishmentRegisterID:   est.EstablishmentRegisterID,
		OfficeID:                  est.OfficeID,
		OfficeName:                est.OfficeName,
		EstablishmentRegisterName: est.EstablishmentRegisterName,
		CreatedBy:                 est.CreatedBy,
		CreatedOn:                 est.CreatedOn,
		Status:                    est.Status,
	}
}

// API response struct for successful creation
type CreateEstablishmentRegisterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      CreateEstablishmentRegisterResponse `json:"data"`
}

// Response struct for post mapping detail creation
type CreatePostMappingDetailResponse struct {
	EmployeePostID int `json:"employee_post_id"`
	OfficeID       int `json:"employee_office_id"`
}

// Helper function to generate the response struct
func NewCreatePostMappingDetailResponse(postMap *domain.PosttoPostMap) CreatePostMappingDetailResponse {
	return CreatePostMappingDetailResponse{
		EmployeePostID: postMap.EmployeePostID,
		OfficeID:       postMap.OfficeID,
	}
}

// API response struct for post mapping creation
type CreatePostMappingDetailAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      CreatePostMappingDetailResponse `json:"data"`
}

// Response struct for updating array of EmpPostID for a particular field
type UpdateArrayOfEmpPostIDResponse struct {
	EmployeePostID int         `json:"employee_post_id"`
	UpdatedDate    time.Time   `json:"updated_date"`
	FieldName      string      `json:"field_name"`
	FieldValue     interface{} `json:"field_value"`
}

// Helper function to generate the response array
func NewPostMapUpdateResponseArray(
	postMaps []domain.PosttoPostMap, fieldName string, fieldValue interface{},
) []UpdateArrayOfEmpPostIDResponse {
	var responses []UpdateArrayOfEmpPostIDResponse
	for _, postMap := range postMaps {
		response := UpdateArrayOfEmpPostIDResponse{
			EmployeePostID: postMap.EmployeePostID,
			UpdatedDate:    postMap.UpdatedDate,
			FieldName:      fieldName,
			FieldValue:     fieldValue,
		}
		responses = append(responses, response)
	}
	return responses
}

// API response struct for updating post mapping details
type UpdateArrayOfEmpPostIDAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []UpdateArrayOfEmpPostIDResponse `json:"data"`
}

// Response struct for updating multiple fields for EmpPostID
type UpdateArrayOfEmpPostIDForManyFieldResponse struct {
	EmployeePostID int         `json:"employee_post_id"`
	UpdatedDate    time.Time   `json:"updated_date"`
	FieldUpdated   string      `json:"field_updated"`
	NewValue       interface{} `json:"new_value"`
}

// Helper function to generate the response array
func NewPostMapUpdateResponseArrayForMultipleFields(
	postMaps []domain.PosttoPostMap,
) []UpdateArrayOfEmpPostIDForManyFieldResponse {
	var responses []UpdateArrayOfEmpPostIDForManyFieldResponse
	for _, postMap := range postMaps {
		response := UpdateArrayOfEmpPostIDForManyFieldResponse{
			EmployeePostID: postMap.EmployeePostID,
			UpdatedDate:    postMap.UpdatedDate,
			FieldUpdated:   postMap.FieldUpdated,
			NewValue:       postMap.NewValue,
		}
		responses = append(responses, response)
	}
	return responses
}

// API response struct for updating multiple fields
type UpdateArrayOfEmpPostIDForManyFieldAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []UpdateArrayOfEmpPostIDForManyFieldResponse `json:"data"`
}

// Response struct for creating Post Mapping Detail Maker
type CreatePostMappingDetailMakerResponse struct {
	EmployeePostID   int         `json:"employee_post_id"`
	EmployeeOfficeID int         `json:"employee_office_id"`
	ApproveStatus    string      `json:"approve_status"`
	CreatedDate      time.Time   `json:"created_date"`
	CreatedBy        string      `json:"created_by"`
	FieldUpdated     string      `json:"field_updated"`
	NewValue         interface{} `json:"new_value"`
}

// Helper function to generate the response array
func NewPostMapCreateResponseArray(
	postMaps []domain.PosttoPostMap,
) []CreatePostMappingDetailMakerResponse {
	var responses []CreatePostMappingDetailMakerResponse
	for _, postMap := range postMaps {
		response := CreatePostMappingDetailMakerResponse{
			EmployeePostID:   postMap.EmployeePostID,
			EmployeeOfficeID: postMap.OfficeID,
			ApproveStatus:    postMap.ApproveStatus,
			CreatedDate:      postMap.UpdatedDate,
			CreatedBy:        "user", // Set as needed
			FieldUpdated:     postMap.FieldUpdated,
			NewValue:         postMap.NewValue,
		}
		responses = append(responses, response)
	}
	return responses
}

// API response struct for creating Post Mapping Detail Maker
type CreatePostMappingDetailMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []CreatePostMappingDetailMakerResponse `json:"data"`
}

// Response struct for approving post mapping detail maker
type ApprovePostMappingDetailMakerResponse struct {
	Message string `json:"message"`
}

// Helper function to generate the response array
func NewApprovePostMappingDetailMakerResponse(mesage string) ApprovePostMappingDetailMakerResponse {
	return ApprovePostMappingDetailMakerResponse{
		Message: mesage,
	}
}

// API response struct for approving post mapping detail maker
type ApprovePostMappingDetailMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      ApprovePostMappingDetailMakerResponse `json:"data"`
}

type PostManagementDeleteResponse struct {
	Message string `json:"message"`
}

func NewPostManagementDeleteResponse(successMsg string) PostManagementDeleteResponse {
	return PostManagementDeleteResponse{
		Message: successMsg,
	}
}

type PostManagementDeleteAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      PostManagementDeleteResponse `json:"data"`
}

type UpdatePostMappingDetailResponse struct {
	EmployeePostID int                    `json:"employee_post_id"`
	UpdatedFields  map[string]interface{} `json:"updated_fields"`
	UpdatedDate    time.Time              `json:"updated_date"`
}

// NewUpdatePostMappingDetailResponse constructs the response for the updated post mapping details.
func NewUpdatePostMappingDetailResponse(updatedPost *domain.PosttoPostMap) UpdatePostMappingDetailResponse {
	// Construct the response by mapping the updated fields and the updated date
	return UpdatePostMappingDetailResponse{
		EmployeePostID: updatedPost.EmployeePostID,
		UpdatedFields: map[string]interface{}{ // Include all possible updated fields with their new values
			"gds_leave_sanc_authority_1":      updatedPost.GDSLeaveSancAuthority1,
			"gds_leave_sanc_authority_2":      updatedPost.GDSLeaveSancAuthority2,
			"reporting_authority":             updatedPost.ReportingAuthority,
			"apar_reporting_authority":        updatedPost.AparReportingAuthority,
			"apar_review_authority":           updatedPost.AparReviewAuthority,
			"apar_accepting_authority":        updatedPost.AparAcceptingAuthority,
			"apar_represent_authority":        updatedPost.AparRepresentAuthority,
			"service_book_approve_authority1": updatedPost.ServiceBookApproveAuthority1,
			"service_book_approve_authority2": updatedPost.ServiceBookApproveAuthority2,
			"leave_sanc_authority_1":          updatedPost.LeaveSancAuthority1,
			"leave_sanc_authority_2":          updatedPost.LeaveSancAuthority2,
			"leave_sanc_authority_3":          updatedPost.LeaveSancAuthority3,
			"pay_approve_authority1":          updatedPost.PayApproveAuthority1,
			//"pay_approve_authority2":          updatedPost.PayApproveAuthority2,
			"leave_fwd_authority1":           updatedPost.LeaveFWDAuthority1,
			"leave_fwd_authority2":           updatedPost.LeaveFWDAuthority2,
			"pay_fwd_authority1":             updatedPost.PayFWDAuthority1,
			"pay_fwd_authority2":             updatedPost.PayFWDAuthority2,
			"appointing_authority":           updatedPost.AppointingAuthority,
			"disciplinary_authority":         updatedPost.DisciplinaryAuthority,
			"ddo_authority":                  updatedPost.DdoAuthority,
			"admin_authority":                updatedPost.AdminAuthority,
			"pension_sanctioning_authority":  updatedPost.PensionSanctioningAuthority,
			"pension_authorising_authority":  updatedPost.PensionAuthorisingAuthority,
			"service_book_foward_authority1": updatedPost.ServiceBookForwardAuthority1,
			"service_book_foward_authority2": updatedPost.ServiceBookForwardAuthority2,
			"role_authority":                 updatedPost.RoleAuthority,
		},
		UpdatedDate: updatedPost.UpdatedDate,
	}
}

type UpdatePostMappingDetailAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:"meta_data"`
	Data                      UpdatePostMappingDetailResponse `json:"data"`
}

type UploadFileResponse struct {
	OfficeID             int       `json:"office_id"`
	DocumentName         string    `json:"document_name"`
	DocumentType         string    `json:"document_type"`
	DocumentSize         int64     `json:"document_size"`
	DocumentFilePath     string    `json:"document_file_path"`
	DocumentUploadStatus string    `json:"document_upload_status"`
	DocumentUploadedBy   string    `json:"document_uploaded_by"`
	DocumentUploadedDate time.Time `json:"document_uploaded_date"`
}

// NewUploadFileResponse creates the response for the uploaded file.
func NewUploadFileResponse(document domain.Document) UploadFileResponse {
	return UploadFileResponse{
		OfficeID:             document.OfficeID,
		DocumentName:         document.DocumentName,
		DocumentType:         document.DocumentType,
		DocumentSize:         document.DocumentSize,
		DocumentFilePath:     document.DocumentFilePath,
		DocumentUploadStatus: document.DocumentUploadStatus,
		DocumentUploadedBy:   document.DocumentUploadedBy,
		DocumentUploadedDate: document.DocumentUploadedDate,
	}
}

type UploadFileAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      UploadFileResponse `json:"data"`
}

type DownloadFilesResponse struct {
	OfficeID       int      `json:"office_id"`
	FileNames      []string `json:"file_names"`
	ZipFileName    string   `json:"zip_file_name"`
	TotalFiles     int      `json:"total_files"`
	DownloadStatus string   `json:"download_status"`
}

// NewDownloadFilesResponse constructs the DownloadFilesResponse.
func NewDownloadFilesResponse(officeID int, fileNames []string, zipFileName string, downloadStatus string) DownloadFilesResponse {
	return DownloadFilesResponse{
		OfficeID:       officeID,
		FileNames:      fileNames,
		ZipFileName:    zipFileName,
		TotalFiles:     len(fileNames),
		DownloadStatus: downloadStatus,
	}
}

type DownloadFilesAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      DownloadFilesResponse `json:"data"`
}

// ApprovePostManagementMakerForExchangePostResponse represents the response structure
// for the approval of post management maker for exchange post.
type ApprovePostManagementMakerForExchangePostResponse struct {
	ApprovedPostIDs []int  `json:"approvedPostIDs"`
	ApprovedBy      string `json:"approvedBy"`
	Status          string `json:"status"`
	Message         string `json:"message"`
}

// NewApprovePostManagementMakerForExchangePostResponse creates a new response for the ApprovePostManagementMakerForExchangePost API.
func NewApprovePostManagementMakerForExchangePostResponse(postIDs []int, approvedBy string) ApprovePostManagementMakerForExchangePostResponse {
	return ApprovePostManagementMakerForExchangePostResponse{
		ApprovedPostIDs: postIDs,
		ApprovedBy:      approvedBy,
		Status:          "Success",
		Message:         "Post management maker records approved successfully for exchange post",
	}
}

// ApprovePostManagementMakerForExchangePostAPIResponse is the API response structure.
type ApprovePostManagementMakerForExchangePostAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      ApprovePostManagementMakerForExchangePostResponse `json:"data"`
}

type GetMulAuthIDResponse struct {
	EmployeePostID        int    `json:"employee_post_id"`
	PostMapID             string `json:"post_map_id"`
	PostMapPostId         int32  `json:"post_map_post_id"`
	PostMappingColumnName string `json:"post_map_column_name"`
	OfficeId              int    `json:"post_map_office_id"`
}

// Helper function to convert the domain model to the response struct
func NewGetMulAuthIDResponse(data domain.PosttoPostMap) GetMulAuthIDResponse {
	return GetMulAuthIDResponse{
		EmployeePostID:        data.EmployeePostID,
		PostMapID:             data.PostMapID,
		PostMapPostId:         data.PostMapPostId,
		PostMappingColumnName: data.PostMappingColumnName,
		OfficeId:              data.OfficeID,
	}
}

type GetMulAuthIDAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	TotalRecords              int                    `json:"total_records"`
	Data                      []GetMulAuthIDResponse `json:"data"`
}

type ApprovePostManagementMasterWithMakerResponse struct {
	Message string `json:"message"`
}

func NewApprovePostManagementMasterWithMakerResponse(message string) ApprovePostManagementMasterWithMakerResponse {
	return ApprovePostManagementMasterWithMakerResponse{
		Message: message,
	}
}

type ApprovePostManagementMasterWithMakerAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      ApprovePostManagementMasterWithMakerResponse `json:"data"`
}

type GetPostMappingMakerDetailsResponse struct {
	EmployeePostID int    `json:"employee_post_id"`
	OfficeID       int    `json:"office_id"`
	OfficeName     string `json:"office_name"`
	PostName       string `json:"post_name"`
	// ApproveStatus           string    `json:"approve_status"`
	// VigilanceMakerAuthority int32     `json:"vigilance_maker_authority"`
	// AdminOffice             int32     `json:"admin_office"`
	// UpdatedDate             time.Time `json:"updated_date"`
	// UpdatedBy               string    `json:"updated_by"`
	CreatedDate time.Time `json:"created_date"`
	CreatedBy   string    `json:"created_by"`
}

func NewGetPostMappingMakerDetailsResponse(
	data []domain.PosttoPostMap,
) []GetPostMappingMakerDetailsResponse {
	var response []GetPostMappingMakerDetailsResponse
	for _, details := range data {
		resp := GetPostMappingMakerDetailsResponse{
			EmployeePostID: details.EmployeePostID,
			OfficeID:       details.OfficeID,
			OfficeName:     details.OfficeName,
			PostName:       details.PostName,
			// ApproveStatus:           details.ApproveStatus,
			// VigilanceMakerAuthority: details.VigilenceMakerAuthority, // Correct type
			// AdminOffice:             details.AdminOffice,             // Correct type
			// UpdatedDate:             details.UpdatedDate,
			// UpdatedBy:               details.UpdatedBy,
			CreatedDate: details.CreatedDate,
			CreatedBy:   details.CreatedBy,
		}
		response = append(response, resp)
	}
	return response
}

type GetPostMappingMakerDetailsAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []GetPostMappingMakerDetailsResponse `json:"data"`
}

type ListManagementMakerResponse struct {
	DivisionName            string `json:"division_name" db:"division_name"`
	DivisionOfficeID        int    `json:"division_office_id" db:"division_office_id"`
	SubDivisionName         string `json:"sub_division_name" db:"sub_division_name"`
	SubDivisionOfficeID     int    `json:"sub_division_office_id" db:"sub_division_office_id"`
	OfficeName              string `json:"office_name" db:"office_name"`
	OfficeID                int    `json:"office_id" db:"office_id"`
	PostID                  int    `json:"post_id" db:"post_id"`
	PostName                string `json:"post_name" db:"post_name"`
	GroupID                 int    `json:"group_id" db:"group_id"`
	EmployeeGroup           string `json:"employee_group" db:"employee_group"`
	CadreID                 int    `json:"cadre_id" db:"cadre_id"`
	CadreName               string `json:"cadre_name" db:"cadre_name"`
	FilledStatus            string `json:"filled_status" db:"filled_status"`
	EstablishmentRegisterID int    `json:"establishment_register_id" db:"establishment_register_id"`
	Designation             string `json:"designation" db:"designation"`
	PayLevel                int    `json:"pay_level" db:"pay_level"`
	GradePay                int    `json:"grade_pay" db:"grade_pay"`
	UpdateDate              string `json:"updated_date" db:"updated_date"`
	HoName                  string `json:"ho_name" db:"ho_name"`
	HoID                    int    `json:"ho_id" db:"ho_id"`
}

func NewPostManagementMaker(res []*domain.ListManagementMaker) []ListManagementMakerResponse {
	var rsp []ListManagementMakerResponse
	for _, req := range res {
		rsp = append(rsp, ListManagementMakerResponse{
			DivisionName:            req.DivisionName.String,
			DivisionOfficeID:        req.DivisionOfficeID.Int,
			SubDivisionName:         req.SubDivisionName.String,
			SubDivisionOfficeID:     req.SubDivisionOfficeID.Int,
			OfficeName:              req.OfficeName.String,
			OfficeID:                req.OfficeID.Int,
			PostID:                  req.PostID.Int,
			PostName:                req.PostName.String,
			EmployeeGroup:           req.EmployeeGroup.String,
			CadreID:                 req.CadreID.Int,
			CadreName:               req.CadreName.String,
			FilledStatus:            req.FilledStatus.String,
			EstablishmentRegisterID: req.EstablishmentRegisterID.Int,
			Designation:             req.Designation.String,
			PayLevel:                req.PayLevel.Int,
			GradePay:                req.GradePay.Int,
			UpdateDate:              req.UpdateDate.String,
			HoName:                  req.HoName.String,
			HoID:                    req.HoID.Int,
		})
	}
	return rsp
}

type FetchPostManagementresponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListManagementMakerResponse `json:"data"`
}

type ListVacantPostsResponse struct {
	OfficeID      int    `json:"office_id" db:"office_id"`
	OfficeName    string `json:"office_name" db:"office_name"`
	PostID        int    `json:"post_id" db:"post_id"`
	PostName      string `json:"post_name" db:"post_name"`
	GroupID       int    `json:"group_id" db:"group_id"`
	CadreName     string `json:"cadre_name" db:"cadre_name"`
	Designation   string `json:"designation" db:"designation"`
	EmployeeGroup string `json:"employee_group" db:"employee_group"`
	PostStatus    string `json:"post_status" db:"post_status"`
}

func NewListVacantPostsMaker(res []*domain.ListAvailablePosts) []ListVacantPostsResponse {
	var rsp []ListVacantPostsResponse
	for _, req := range res {
		rsp = append(rsp, ListVacantPostsResponse{
			OfficeName:    req.OfficeName.String,
			OfficeID:      req.OfficeID.Int,
			PostID:        req.PostID.Int,
			PostName:      req.PostName.String,
			EmployeeGroup: req.EmployeeGroup.String,
			CadreName:     req.CadreName.String,
			Designation:   req.Designation.String,
			GroupID:       req.GroupID.Int,
			PostStatus:    req.PostStatus.String,
		})
	}
	return rsp
}

type ListVacantPostsresponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListVacantPostsResponse `json:"data"`
}

type ListAvailablePostsResponse struct {
	OfficeID      int    `json:"office_id" db:"office_id"`
	OfficeName    string `json:"office_name" db:"office_name"`
	PostID        int    `json:"post_id" db:"post_id"`
	PostName      string `json:"post_name" db:"post_name"`
	GroupID       int    `json:"group_id" db:"group_id"`
	CadreName     string `json:"cadre_name" db:"cadre_name"`
	Designation   string `json:"designation" db:"designation"`
	EmployeeGroup string `json:"employee_group" db:"employee_group"`
	PostStatus    string `json:"post_status" db:"post_status"`
	FilledStatus  string `json:"filled_status" db:"filled_status"`
}

func NewListAvailablePostsMaker(res []*domain.ListAvailablePosts) []ListAvailablePostsResponse {
	var rsp []ListAvailablePostsResponse
	for _, req := range res {
		rsp = append(rsp, ListAvailablePostsResponse{
			OfficeName:    req.OfficeName.String,
			OfficeID:      req.OfficeID.Int,
			PostID:        req.PostID.Int,
			PostName:      req.PostName.String,
			EmployeeGroup: req.EmployeeGroup.String,
			CadreName:     req.CadreName.String,
			Designation:   req.Designation.String,
			GroupID:       req.GroupID.Int,
			PostStatus:    req.PostStatus.String,
			FilledStatus:  req.FilledStatus.String,
		})
	}
	return rsp
}

type ListAvailablePostsresponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListAvailablePostsResponse `json:"data"`
}
type CadreListResponse struct {
	StatusCodeAndMessage port.StatusCodeAndMessage `json:"status"`
	MetaDataResponse     port.MetaDataResponse     `json:"metadata"`
	Data                 []domain.CadreMasterNew   `json:"data"`
}
type DesignationListResponse struct {
	port.StatusCodeAndMessage `json:"status"`
	port.MetaDataResponse     `json:"metadata"`
	Data                      []domain.DesignationMasterNew `json:"data"`
}

type ListGroupMasterResponse struct {
	GroupID   int    `json:"group_id"`
	GroupName string `json:"group_name"`
}

type ListGroupAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListGroupMasterResponse `json:"data"`
}

func NewListGroupMasterMaker(res []*domain.ListGroupMaster) []ListGroupMasterResponse {
	var rsp []ListGroupMasterResponse
	for _, req := range res {
		rsp = append(rsp, ListGroupMasterResponse{
			GroupID:   req.GroupID,
			GroupName: req.GroupName,
		})
	}
	return rsp
}

func safeString(s *string) string {
	if s != nil {
		return *s
	}
	return ""
}

type ListOfficeDetailsResponse struct {
	CadreName                string `json:"cadre_name"`
	GroupName                string `json:"group_name"`
	PostID                   string `json:"post_id"`
	Designation              string `json:"designation"`
	OfficeID                 string `json:"office_id"`
	DivisionOfficeID         string `json:"division_office_id"`
	CircleOfficeID           string `json:"circle_office_id"`
	RegionOfficeID           string `json:"region_office_id"`
	ReportingOfficeID        string `json:"reporting_office_id"`
	ReportingAuthorityPostID string `json:"reporting_authority_post_id"`
	CadreID                  string `json:"cadre_id"`
	DesignationID            string `json:"designation_id"`
}

func NewListOfficeDetailsMaker(res []*domain.ListOfficeDetails) []ListOfficeDetailsResponse {
	var rsp []ListOfficeDetailsResponse
	for _, req := range res {
		rsp = append(rsp, ListOfficeDetailsResponse{
			CadreName:                safeString(req.CadreName),
			GroupName:                safeString(req.GroupName),
			PostID:                   safeString(req.PostID),
			Designation:              safeString(req.Designation),
			OfficeID:                 req.OfficeID,
			DivisionOfficeID:         safeString(req.DivisionOfficeID),
			CircleOfficeID:           safeString(req.CircleOfficeID),
			RegionOfficeID:           safeString(req.RegionOfficeID),
			ReportingOfficeID:        safeString(req.ReportingOfficeID),
			ReportingAuthorityPostID: safeString(req.ReportingAuthorityPostID),
			CadreID:                  safeString(req.CadreID),
			DesignationID:            safeString(req.DesignationID),
		})
	}
	return rsp
}

type ListOfficeDetailsAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListOfficeDetailsResponse `json:"data"`
}

type ListGroupCadreResponse struct {
	CadreID   string `json:"cadre_id"`
	CadreName string `json:"cadre_name"`
	PayLevel  string `json:"pay_level"`
	GradePay  string `json:"grade_pay"`
	GroupID   string `json:"group_id"`
	GroupName string `json:"group_name"`
}

func NewListGroupCadreMaker(res []*domain.ListGroupCadre) []ListGroupCadreResponse {
	var rsp []ListGroupCadreResponse
	for _, req := range res {
		rsp = append(rsp, ListGroupCadreResponse{
			CadreName: safeString(req.CadreName),
			GroupName: safeString(req.GroupName),
			CadreID:   safeString(req.CadreID),
			PayLevel:  safeString(req.PayLevel),
			GradePay:  safeString(req.GradePay),
			GroupID:   req.GroupID,
		})
	}
	return rsp
}

type ListGroupCadreAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListGroupCadreResponse `json:"data"`
}

type ListDesignationByCadreResponse struct {
	DesignationID int    `json:"designation_id"`
	Designation   string `json:"designation"`
	CadreID       int    `json:"cadre_id"`
	CadreName     string `json:"cadre_name"`
}

func NewListDesignationByCadreResponse(data []domain.DesignationMasterNew) []ListDesignationByCadreResponse {
	var response []ListDesignationByCadreResponse
	for _, designation := range data {
		designationResponse := ListDesignationByCadreResponse{
			DesignationID: designation.DesignationID,
			Designation:   designation.Designation,
		}
		response = append(response, designationResponse)
	}
	return response
}

type ListDesignationByCadreAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListAllCadresResponse `json:"data"`
}

type CreateCadreMasterResponse struct {
	CadreID int `json:"cadre_id"`
}

func NewCreateCadreMasterResponse(data domain.CadreMaster) CreateCadreMasterResponse {
	response := CreateCadreMasterResponse{
		CadreID: int(data.CadreID.Int32),
	}
	return response
}

type CreateCadreMasterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      CreateCadreMasterResponse `json:"data"`
}

type CreateDesignationMasterResponse struct {
	DesignationID  int `json:"designation_id"`
	DesignationUID int `json:"designation_uid"`
}

func NewCreateDesignationMasterResponse(data domain.DesignationMaster) CreateDesignationMasterResponse {
	response := CreateDesignationMasterResponse{
		DesignationID:  data.DesignationID,
		DesignationUID: data.DesignationUID,
	}
	return response
}

type CreateDesignationMasterAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      CreateDesignationMasterResponse `json:"data"`
}
type ListAllDesignationsResponse struct {
	DesignationID  int       `json:"designation_id" db:"designation_id"`
	Designation    string    `json:"designation" db:"designation"`
	GroupName      string    `json:"group_name" db:"group_name"`
	CadreName      string    `json:"cadre_name" db:"cadre_name"`
	ValidFrom      time.Time `json:"valid_from" db:"valid_from"`
	ValidTo        time.Time `json:"valid_to" db:"valid_to"`
	Status         string    `json:"status" db:"status"`
	Remarks        string    `json:"remarks" db:"remarks"`
	CadreId        int       `json:"cadre_id" db:"cadre_id"`
	GroupId        int16     `json:"group_id" db:"group_id"`
	DesignationUID int       `json:"designation_uuid" db:"designation_uuid"`
}

func NewListAllDesignationsResponse(data []domain.DesignationMaster) []ListAllDesignationsResponse {
	var response []ListAllDesignationsResponse
	for _, designation := range data {
		allDesignationResponse := ListAllDesignationsResponse{
			DesignationID:  designation.DesignationID,
			Designation:    designation.Designation,
			GroupName:      designation.GroupName,
			CadreName:      designation.CadreName,
			ValidFrom:      designation.ValidFrom,
			ValidTo:        designation.ValidTo,
			Status:         designation.Status,
			Remarks:        designation.Remarks,
			CadreId:        designation.CadreId,
			GroupId:        designation.GroupId,
			DesignationUID: designation.DesignationUID,
		}
		response = append(response, allDesignationResponse)
	}
	return response
}

type ListAllDesignationsAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListAllDesignationsResponse `json:"data"`
}

type PostManagementByOfficeIDResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      string
}

type VacantPostResponse struct {
	PostID int64 `json:"post_id"`
}

func NewPostManagementByOfficeIDResponse(data []domain.VacantPost) []VacantPostResponse {
	var response []VacantPostResponse
	for _, details := range data {
		resp := VacantPostResponse{
			PostID: details.PostID,
		}
		response = append(response, resp)
	}
	return response
}

type PostManagementByOfficeIDResponse1 struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []VacantPostResponse `json:"data"`
}

type PostDetailsResponse struct {
	OfficeID          int64  `json:"office_id"`
	OfficeName        string `json:"office_name"`
	PostID            int64  `json:"post_id"`
	PostName          string `json:"post_name"`
	GroupID           int64  `json:"group_id"`
	GroupName         string `json:"group_name"`
	CadreID           int64  `json:"cadre_id"`
	CadreName         string `json:"cadre_name"`
	Designation       string `json:"designation"`
	DesignationID     int64  `json:"designation_id"`
	IsHeadOfTheOffice bool   `json:"is_head_of_the_office"`
	EmployeeID        int64  `json:"employee_id"`
	EmployeeName      string `json:"employee_name"`
}

func NewPostDetailsResponse(data []domain.PostDetails) []PostDetailsResponse {
	var response []PostDetailsResponse
	for _, details := range data {
		resp := PostDetailsResponse{
			OfficeID:          details.OfficeID,
			OfficeName:        details.OfficeName.String,
			PostID:            details.PostID,
			PostName:          details.PostName.String,
			GroupID:           details.GroupID.Int64,
			GroupName:         details.GroupName.String,
			CadreID:           details.CadreID.Int64,
			CadreName:         details.CadreName.String,
			Designation:       details.Designation.String,
			DesignationID:     details.DesignationID.Int64,
			IsHeadOfTheOffice: details.IsHeadOfTheOffice.Bool,
			EmployeeID:        details.EmployeeID.Int64,
			EmployeeName:      details.EmployeeName.String,
		}
		response = append(response, resp)
	}
	return response
}

type PostDetailsAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []PostDetailsResponse `json:"data"`
}

func NewFetchPostsByOfficeIDResponse2(
	masters []domain.PostManagementMasterNew,
	makers []domain.PostManagementMaker,
) []FetchPostsByOfficeIDResponseNew {
	var response []FetchPostsByOfficeIDResponseNew

	// for _, item := range masters {
	// 	resp := FetchPostsByOfficeIDResponseNew{
	// 		OfficeID:   int(item.OfficeID.Int32),
	// 		PostID:     int(item.PostID.Int32),
	// 		PostName:   item.PostName.String,
	// 		OfficeName: item.OfficeName.String,
	// 		GroupId:    int(item.GroupId.Int32),
	// 		// GroupName:                 item.GroupName.String,
	// 		GroupName: func() string {
	// 			if item.GroupName.Valid {
	// 				return item.GroupName.String
	// 			}
	// 			return ""
	// 		}(),
	// 		CadreID:                   int(item.CadreID.Int32),
	// 		CadreName:                 item.CadreName.String,
	// 		DesignationID:             int(item.DesignationId.Int32),
	// 		Designation:               item.Designation.String,
	// 		FilledStatus:              item.FilledStatus.String,
	// 		PostStatus:                item.PostStatus.String,
	// 		Status:                    item.Status.String,
	// 		PermanentStatus:           item.PermanentStatus.Bool,
	// 		AllowancesAttached:        item.AllowancesAttached.Bool,
	// 		AllowanceDescription:      item.AllowanceDescription.String,
	// 		CreatedBy:                 item.CreatedBy.String,
	// 		CreatedOn:                 item.CreatedOn.Time,
	// 		ApprovedBy:                item.ApprovedBy.String,
	// 		ApprovedOn:                item.ApprovedOn.Time,
	// 		UpdatedBy:                 item.UpdatedBy.String,
	// 		UpdatedOn:                 item.UpdatedOn.Time,
	// 		ValidFrom:                 item.ValidFrom.Time,
	// 		ValidTo:                   item.ValidTo.Time,
	// 		OrderCaseMark:             item.OrderCaseMark.String,
	// 		OrderDate:                 item.OrderDate.Time,
	// 		UploadOrderDocName:        item.UploadOrderDocName.String,
	// 		EstablishmentRegisterID:   int(item.EstablishmentRegisterID.Int32),
	// 		EstablishmentRegisterName: item.EstablishmentRegisterName.String,
	// 		EmployeeGroup:             item.EmployeeGroup.String,
	// 		SanctionedStrength:        int(item.SanctionedStrength.Int32),
	// 		Remarks:                   item.Remarks.String,
	// 		ApproveStatus:             item.ApproveStatus.String,
	// 		IsHeadOfTheOffice:         item.IsHeadOfTheOffice.Bool,
	// 		EmployeeID:                item.EmployeeID.Int32,
	// 		EmployeeName:              item.EmployeeName.String,
	// 		// New authority fields
	// 		EmployeePostID:          item.EmployeePostID.Int32,
	// 		LeaveSancAuthority1:     item.LeaveSancAuthority1.String,
	// 		LeaveSancAuthority2:     item.LeaveSancAuthority2.String,
	// 		PayApproveAuthority1:    item.PayApproveAuthority1.String,
	// 		AppointingAuthority:     item.AppointingAuthority.String,
	// 		DisciplinaryAuthority:   item.DisciplinaryAuthority.String,
	// 		DDOAuthority:            item.DDOAuthority.String,
	// 		EmployeeOfficeID:        item.EmployeeOfficeID.Int32,
	// 		VigilanceMakerAuthority: item.VigilanceMakerAuthority.String,
	// 	}
	// 	response = append(response, resp)
	// }

	for _, item := range masters {
		resp := FetchPostsByOfficeIDResponseNew{
			OfficeID:                  item.OfficeID,
			PostID:                    item.PostID,
			PostName:                  item.PostName,
			OfficeName:                item.OfficeName,
			GroupId:                   item.GroupId,
			GroupName:                 item.GroupName,
			CadreID:                   item.CadreID,
			CadreName:                 item.CadreName,
			DesignationID:             item.DesignationId,
			Designation:               item.Designation,
			FilledStatus:              item.FilledStatus,
			PostStatus:                item.PostStatus,
			Status:                    item.Status,
			PermanentStatus:           item.PermanentStatus,
			AllowancesAttached:        item.AllowancesAttached,
			AllowanceDescription:      item.AllowanceDescription,
			CreatedBy:                 item.CreatedBy,
			CreatedOn:                 item.CreatedDate,
			ApprovedBy:                item.ApprovedBy,
			ApprovedOn:                item.ApprovedDate,
			UpdatedBy:                 item.UpdatedBy,
			UpdatedOn:                 item.UpdatedDate,
			ValidFrom:                 item.ValidFrom,
			ValidTo:                   item.ValidTo,
			OrderCaseMark:             item.OrderCaseMark,
			OrderDate:                 item.OrderDate,
			UploadOrderDocName:        item.UploadOrderDocName,
			EstablishmentRegisterID:   item.EstablishmentRegisterID,
			EstablishmentRegisterName: item.EstablishmentRegisterName,
			EmployeeGroup:             item.EmployeeGroup,
			SanctionedStrength:        item.SanctionedStrength,
			Remarks:                   item.Remarks,
			ApproveStatus:             item.ApproveStatus,
			IsHeadOfTheOffice:         item.IsHeadOfTheOffice,
			EmployeeID:                item.EmployeeID,
			EmployeeName:              item.EmployeeName,
			// New authority fields
			EmployeePostID:          item.EmployeePostID,
			LeaveSancAuthority1:     item.LeaveSancAuthority1,
			LeaveSancAuthority2:     item.LeaveSancAuthority2,
			PayApproveAuthority1:    item.PayApproveAuthority1,
			AppointingAuthority:     item.AppointingAuthority,
			DisciplinaryAuthority:   item.DisciplinaryAuthority,
			DDOAuthority:            item.DDOAuthority,
			EmployeeOfficeID:        item.EmployeeOfficeID,
			VigilanceMakerAuthority: item.VigilanceMakerAuthority,
		}
		response = append(response, resp)
	}
	for _, item := range makers {
		resp := FetchPostsByOfficeIDResponseNew{
			OfficeID:                  item.OfficeID,
			PostID:                    item.PostID,
			PostName:                  item.PostName,
			OfficeName:                item.OfficeName,
			GroupId:                   item.GroupId,
			GroupName:                 item.GroupName,
			CadreID:                   item.CadreID,
			CadreName:                 item.CadreName,
			DesignationID:             item.DesignationId,
			Designation:               item.Designation,
			FilledStatus:              item.FilledStatus,
			PostStatus:                item.PostStatus,
			Status:                    item.Status,
			PermanentStatus:           item.PermanentStatus,
			AllowancesAttached:        item.AllowancesAttached,
			AllowanceDescription:      item.AllowanceDescription,
			CreatedBy:                 item.CreatedBy,
			CreatedOn:                 item.CreatedDate,
			ValidFrom:                 item.ValidFrom,
			ValidTo:                   item.ValidTo,
			OrderCaseMark:             item.OrderCaseMark,
			OrderDate:                 item.OrderDate,
			UploadOrderDocName:        item.UploadOrderDocName,
			EstablishmentRegisterID:   item.EstablishmentRegisterID,
			EstablishmentRegisterName: item.EstablishmentRegisterName,
			EmployeeGroup:             item.EmployeeGroup,
			SanctionedStrength:        item.SanctionedStrength,
			ExchangePostID:            item.ExchangePostID,
			Remarks:                   item.Remarks,
			ApproveStatus:             item.ApproveStatus,
			ApprovePostID:             item.ApprovePostID,
		}
		response = append(response, resp)
	}

	return response
}

type FetchPostsByOfficeIDResponseNew struct {
	OfficeID                  int       `json:"office_id"`
	PostID                    int       `json:"post_id"`
	PostName                  string    `json:"post_name"`
	OfficeName                string    `json:"office_name"`
	GroupId                   int       `json:"group_id"`
	GroupName                 string    `json:"group_name"`
	CadreID                   int       `json:"cadre_id"`
	CadreName                 string    `json:"cadre_name"`
	DesignationID             int       `json:"designation_id"`
	Designation               string    `json:"designation"`
	FilledStatus              string    `json:"filled_status"`
	PostStatus                string    `json:"post_status"`
	Status                    string    `json:"status"`
	PermanentStatus           bool      `json:"permanent_status"`
	AllowancesAttached        bool      `json:"allowances_attached"`
	AllowanceDescription      string    `json:"allowance_description"`
	CreatedBy                 string    `json:"created_by"`
	CreatedOn                 time.Time `json:"created_on"`
	ApprovedBy                string    `json:"approved_by"`
	ApprovedOn                time.Time `json:"approved_on"`
	UpdatedBy                 string    `json:"updated_by"`
	UpdatedOn                 time.Time `json:"updated_on"`
	ValidFrom                 time.Time `json:"valid_from"`
	ValidTo                   time.Time `json:"valid_to"`
	OrderCaseMark             string    `json:"order_casemark"`
	OrderDate                 time.Time `json:"order_date"`
	UploadOrderDocName        string    `json:"upload_order_doc_name"`
	EstablishmentRegisterID   int       `json:"establishment_register_id"`
	EstablishmentRegisterName string    `json:"establishment_register_name"`
	EmployeeGroup             string    `json:"employee_group"`
	SanctionedStrength        int       `json:"sanctioned_strength"`
	Remarks                   string    `json:"remarks"`
	ApproveStatus             string    `json:"approve_status"`
	ApprovePostID             string    `json:"approve_post_id,omitempty"`
	ExchangePostID            int       `json:"exchange_post_id,omitempty"` // Used in maker records
	IsHeadOfTheOffice         bool      `json:"is_head_of_the_office"`

	// Employee Info
	EmployeeID   int32  `json:"employee_id"`
	EmployeeName string `json:"employee_name"`

	// Authority fields from post_mapping_detail
	EmployeePostID          int32  `json:"employee_post_id"`
	LeaveSancAuthority1     string `json:"leave_sanc_authority_1"`
	LeaveSancAuthority2     string `json:"leave_sanc_authority_2"`
	PayApproveAuthority1    string `json:"pay_approve_authority1"`
	AppointingAuthority     string `json:"appointing_authority"`
	DisciplinaryAuthority   string `json:"disciplinary_authority"`
	DDOAuthority            string `json:"ddo_authority"`
	EmployeeOfficeID        int32  `json:"employee_office_id"`
	VigilanceMakerAuthority string `json:"vigilence_maker_authority"`
}

type GetPostRedeplomentByOfficeIDAPIResponse struct {
	StatusCode int                                    `json:"status_code"`
	Message    string                                 `json:"message"`
	Data       []GetPostRedeplomentByOfficeIDResponse `json:"data"`
	TotalCount int                                    `json:"total"`
}

type GetPostRedeplomentByOfficeIDResponse struct {
	OfficeID     null.Int    `json:"office_id"`
	OfficeName   null.String `json:"office_name"`
	OfficeType   null.String `json:"office_type"`
	PostID       null.Int    `json:"post_id"`
	PostName     null.String `json:"post_name"`
	CadreName    null.String `json:"cadre_name"`
	Designation  null.String `json:"designation"`
	EmployeeName null.String `json:"employee_name"`
	FilledStatus null.String `json:"filled_status"`
}

func NewGetPostRedeplomentByOfficeIDResponse(employees []domain.PostRedeployment) []GetPostRedeplomentByOfficeIDResponse {
	rsp := make([]GetPostRedeplomentByOfficeIDResponse, 0, len(employees))
	for _, employee := range employees {
		rsp = append(rsp, GetPostRedeplomentByOfficeIDResponse{
			OfficeID:     employee.OfficeID,
			OfficeName:   employee.OfficeName,
			OfficeType:   employee.OfficeType,
			PostID:       employee.PostID,
			PostName:     employee.PostName,
			CadreName:    employee.CadreName,
			Designation:  employee.Designation,
			EmployeeName: employee.EmployeeName,
			FilledStatus: employee.FilledStatus,
		})
	}
	return rsp
}

type SavePostRedeploymentAPIResponse struct {
	StatusCodeAndMessage port.StatusCodeAndMessage `json:"status_code_and_message"`
	Data                 interface{}               `json:"data,omitempty"`
	Error                string                    `json:"error,omitempty"`
}

type GetCircleOfficeIDsAPIResponse struct {
	StatusCode int                          `json:"status_code"`
	Message    string                       `json:"message"`
	Data       []GetCircleOfficeIDsResponse `json:"data"`
	TotalCount int                          `json:"total"`
}

type GetCircleOfficeIDsResponse struct {
	CircleOfficeID int    `json:"circle_office_id"`
	CircleName     string `json:"circle_name"`
}

func NewGetCircleOfficeIDsResponse(circles []domain.CircleName) []GetCircleOfficeIDsResponse {
	rsp := make([]GetCircleOfficeIDsResponse, 0, len(circles))
	for _, circle := range circles {
		rsp = append(rsp, GetCircleOfficeIDsResponse{
			CircleOfficeID: circle.CircleOfficeID,
			CircleName:     circle.CircleName,
		})
	}
	return rsp
}

type GetRegionalOfficeIDsAPIResponse struct {
	StatusCode int                            `json:"status_code"`
	Message    string                         `json:"message"`
	Data       []GetRegionalOfficeIDsResponse `json:"data"`
	TotalCount int                            `json:"total"`
}

type GetRegionalOfficeIDsResponse struct {
	RegionOfficeID int    `json:"region_office_id"`
	RegionName     string `json:"region_name"`
}

func NewGetRegionalOfficeIDsResponse(regions []domain.RegionName) []GetRegionalOfficeIDsResponse {
	rsp := make([]GetRegionalOfficeIDsResponse, 0, len(regions))
	for _, region := range regions {
		rsp = append(rsp, GetRegionalOfficeIDsResponse{
			RegionOfficeID: region.RegionOfficeID,
			RegionName:     region.RegionName,
		})
	}
	return rsp
}

type GetDivisionalOfficeIDsAPIResponse struct {
	StatusCode int                              `json:"status_code"`
	Message    string                           `json:"message"`
	Data       []GetDivisionalOfficeIDsResponse `json:"data"`
	TotalCount int                              `json:"total"`
}

type GetDivisionalOfficeIDsResponse struct {
	DivisionOfficeID int    `json:"division_office_id"`
	DivisionName     string `json:"division_name"`
}

func NewGetDivisionalOfficeIDsResponse(divisions []domain.DivisionName) []GetDivisionalOfficeIDsResponse {
	rsp := make([]GetDivisionalOfficeIDsResponse, 0, len(divisions))
	for _, division := range divisions {
		rsp = append(rsp, GetDivisionalOfficeIDsResponse{
			DivisionOfficeID: division.DivisionOfficeID,
			DivisionName:     division.DivisionName,
		})
	}
	return rsp
}

type GetCadreDetailsAPIResponse struct {
	StatusCode int                       `json:"status_code"`
	Message    string                    `json:"message"`
	Data       []GetCadreDetailsResponse `json:"data"`
	TotalCount int                       `json:"total"`
}

type GetCadreDetailsResponse struct {
	CadreID   int    `json:"cadre_id"`
	CadreName string `json:"cadre_name"`
}

func NewGetCadreDetailsResponse(cadres []domain.CadreName) []GetCadreDetailsResponse {
	rsp := make([]GetCadreDetailsResponse, 0, len(cadres))
	for _, cadre := range cadres {
		rsp = append(rsp, GetCadreDetailsResponse{
			CadreID:   cadre.CadreID,
			CadreName: cadre.CadreName,
		})
	}
	return rsp
}

type GetPostAndEmployeeHierarchyAPIResponse struct {
	StatusCode int                                   `json:"status_code"`
	Message    string                                `json:"message"`
	Data       []GetPostAndEmployeeHierarchyResponse `json:"data"`
	TotalCount int                                   `json:"total"`
}

type GetPostAndEmployeeHierarchyResponse struct {
	PostID          null.Int    `json:"post_id"`
	PostName        null.String `json:"post_name"`
	OfficeID        null.Int    `json:"office_id"`
	OfficeName      null.String `json:"office_name"`
	GroupID         null.Int    `json:"group_id"`
	GroupName       null.String `json:"group_name"`
	CadreID         null.Int    `json:"cadre_id"`
	CadreName       null.String `json:"cadre_name"`
	DesignationID   null.Int    `json:"designation_id"`
	DesignationName null.String `json:"designation_name"`
	EmployeeID      null.Int    `json:"employee_id"`
	EmployeeName    null.String `json:"employee_name"`
}

func NewGetPostAndEmployeeHierarchyResponse(offices []domain.PostWithEmployee) []GetPostAndEmployeeHierarchyResponse {
	rsp := make([]GetPostAndEmployeeHierarchyResponse, 0, len(offices))
	for _, office := range offices {
		rsp = append(rsp, GetPostAndEmployeeHierarchyResponse{
			PostID:          office.PostID,
			PostName:        office.PostName,
			OfficeID:        office.OfficeID,
			OfficeName:      office.OfficeName,
			GroupID:         office.GroupID,
			GroupName:       office.GroupName,
			CadreID:         office.CadreID,
			CadreName:       office.CadreName,
			DesignationID:   office.DesignationID,
			DesignationName: office.DesignationName,
			EmployeeID:      office.EmployeeID,
			EmployeeName:    office.EmployeeName,
		})
	}
	return rsp
}

type GetPostDetailsbyPostIDAPIResponse struct {
	StatusCode int                              `json:"status_code"`
	Message    string                           `json:"message"`
	Data       []GetPostDetailsbyPostIDResponse `json:"data"`
	TotalCount int                              `json:"total"`
}

type GetPostDetailsbyPostIDResponse struct {
	OfficeID      null.Int    `json:"office_id"`
	OfficeName    null.String `json:"office_name"`
	PostID        null.Int    `json:"post_id" `
	PostName      null.String `json:"post_name" `
	GroupID       null.Int    `json:"group_id" `
	GroupName     null.String `json:"group_name" `
	CadreID       null.Int    `json:"cadre_id" `
	CadreName     null.String `json:"cadre_name" `
	DesignationID null.Int    `json:"designation_id" `
	Designation   null.String `json:"designation_name" `
	PostStatus    null.String `json:"post_status"`
	EmployeeID    null.Int    `json:"employee_id"`
	EmployeeName  null.String `json:"employee_name"`
}

func NewGetPostDetailsbyPostIDResponse(offices []domain.PostIDDetails) []GetPostDetailsbyPostIDResponse {
	rsp := make([]GetPostDetailsbyPostIDResponse, 0, len(offices))
	for _, office := range offices {
		rsp = append(rsp, GetPostDetailsbyPostIDResponse{
			OfficeID:      office.OfficeID,
			OfficeName:    office.OfficeName,
			PostID:        office.PostID,
			PostName:      office.PostName,
			GroupID:       office.GroupID,
			GroupName:     office.GroupName,
			CadreID:       office.CadreID,
			CadreName:     office.CadreName,
			DesignationID: office.DesignationID,
			Designation:   office.Designation,
			PostStatus:    office.PostStatus,
			EmployeeID:    office.EmployeeID,
			EmployeeName:  office.EmployeeName,
		})
	}
	return rsp
}

type GetPostsFilledVacantStatusAPIResponse struct {
	StatusCode int                                  `json:"status_code"`
	Message    string                               `json:"message"`
	Data       []GetPostsFilledVacantStatusResponse `json:"data"`
	TotalCount int                                  `json:"total"`
}

type GetPostsFilledVacantStatusResponse struct {
	CircleOfficeID   null.Int    `json:"circle_office_id"`
	CircleName       null.String `json:"circle_name"`
	RegionOfficeID   null.Int    `json:"region_office_id"`
	RegionName       null.String `json:"region_name"`
	DivisionOfficeID null.Int    `json:"division_office_id"`
	DivisionName     null.String `json:"division_name"`
	OfficeID         null.Int    `json:"office_id"`
	OfficeName       null.String `json:"office_name"`
	GroupID          null.Int    `json:"group_id"`
	GroupName        null.String `json:"group_name"`
	CadreID          null.Int    `json:"cadre_id"`
	CadreName        null.String `json:"cadre_name"`
	TotalPosts       int         `json:"total_posts"`
	TotalFilledPosts int         `json:"total_filled_posts"`
	TotalVacantPosts int         `json:"total_vacant_posts"`
}

func NewGetPostsFilledVacantStatusResponse(posts []domain.PostsStatus) []GetPostsFilledVacantStatusResponse {
	rsp := make([]GetPostsFilledVacantStatusResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetPostsFilledVacantStatusResponse{
			CircleOfficeID:   post.CircleOfficeID,
			CircleName:       post.CircleName,
			RegionOfficeID:   post.RegionOfficeID,
			RegionName:       post.RegionName,
			DivisionOfficeID: post.DivisionOfficeID,
			DivisionName:     post.DivisionName,
			OfficeID:         post.OfficeID,
			OfficeName:       post.OfficeName,
			GroupID:          post.GroupID,
			GroupName:        post.GroupName,
			CadreID:          post.CadreID,
			CadreName:        post.CadreName,
			TotalPosts:       post.TotalPosts,
			TotalFilledPosts: post.TotalFilledPosts,
			TotalVacantPosts: post.TotalVacantPosts,
		})
	}
	return rsp
}

type GetPostsCreatedRedeployedAbolishedAPIResponse struct {
	StatusCode int                                          `json:"status_code"`
	Message    string                                       `json:"message"`
	Data       []GetPostsCreatedRedeployedAbolishedResponse `json:"data"`
	TotalCount int                                          `json:"total"`
}

type GetPostsCreatedRedeployedAbolishedResponse struct {
	CircleOfficeID   null.Int    `json:"circle_office_id"`
	CircleName       null.String `json:"circle_name"`
	RegionOfficeID   null.Int    `json:"region_office_id"`
	RegionName       null.String `json:"region_name"`
	DivisionOfficeID null.Int    `json:"division_office_id"`
	DivisionName     null.String `json:"division_name"`
	OfficeID         null.Int    `json:"office_id"`
	PostsCreated     int         `json:"posts_created"`
	PostsRedeployed  int         `json:"posts_redeployed"`
	PostsAbolished   int         `json:"posts_abolished"`
}

func NewGetPostsCreatedRedeployedAbolishedResponse(posts []domain.PostsCreatedRedeployedAbolished) []GetPostsCreatedRedeployedAbolishedResponse {
	rsp := make([]GetPostsCreatedRedeployedAbolishedResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetPostsCreatedRedeployedAbolishedResponse{
			CircleOfficeID:   post.CircleOfficeID,
			CircleName:       post.CircleName,
			RegionOfficeID:   post.RegionOfficeID,
			RegionName:       post.RegionName,
			DivisionOfficeID: post.DivisionOfficeID,
			DivisionName:     post.DivisionName,
			OfficeID:         post.OfficeID,
			PostsCreated:     post.PostsCreated,
			PostsRedeployed:  post.PostsRedeployed,
			PostsAbolished:   post.PostsAbolished,
		})
	}
	return rsp
}

type GetPostsFilledVacantStatusDetailedAPIResponse struct {
	StatusCode int                                          `json:"status_code"`
	Message    string                                       `json:"message"`
	Data       []GetPostsFilledVacantStatusDetailedResponse `json:"data"`
	TotalCount int                                          `json:"total"`
}

type GetPostsFilledVacantStatusDetailedResponse struct {
	OfficeID      null.Int    `json:"office_id"`
	OfficeName    null.String `json:"office_name"`
	PostID        null.Int    `json:"post_id"`
	PostName      null.String `json:"post_name"`
	GroupID       null.Int    `json:"group_id"`
	GroupName     null.String `json:"group_name"`
	CadreID       null.Int    `json:"cadre_id"`
	CadreName     null.String `json:"cadre_name"`
	DesignationID null.Int    `json:"designation_id"`
	Designation   null.String `json:"designation_name"`
	PostStatus    null.String `json:"post_status"`
	EmployeeID    null.Int    `json:"employee_id"`
	EmployeeName  null.String `json:"employee_name"`
}

func NewGetPostsFilledVacantStatusDetailedResponse(posts []domain.PostsFilledVacantStatusDetailed) []GetPostsFilledVacantStatusDetailedResponse {
	rsp := make([]GetPostsFilledVacantStatusDetailedResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetPostsFilledVacantStatusDetailedResponse{
			OfficeID:      post.OfficeID,
			OfficeName:    post.OfficeName,
			PostID:        post.PostID,
			PostName:      post.PostName,
			GroupID:       post.GroupID,
			GroupName:     post.GroupName,
			CadreID:       post.CadreID,
			CadreName:     post.CadreName,
			DesignationID: post.DesignationID,
			Designation:   post.Designation,
			PostStatus:    post.PostStatus,
			EmployeeID:    post.EmployeeID,
			EmployeeName:  post.EmployeeName,
		})
	}
	return rsp
}

type GetPostDetailsForRedeploymentAPIResponse struct {
	StatusCode int                                     `json:"status_code"`
	Message    string                                  `json:"message"`
	Data       []GetPostDetailsForRedeploymentResponse `json:"data"`
	TotalCount int                                     `json:"total"`
}

type GetPostDetailsForRedeploymentResponse struct {
	OfficeID                  null.Int    `json:"office_id"`
	OfficeName                null.String `json:"office_name"`
	FilledStatus              null.String `json:"filled_status" db:"filled_status"`
	PostStatus                null.String `json:"post_status"`
	AllowancesAttached        *bool       `json:"allowances_attached"`
	AllowanceDescription      null.String `json:"allowance_description"`
	UpdatedBy                 null.String `json:"updated_by"`
	UpdatedDate               *time.Time  `json:"updated_date"`
	Status                    null.String `json:"status"`
	Remarks                   null.String `json:"remarks"`
	ValidFrom                 *time.Time  `json:"valid_from"`
	ValidTo                   *time.Time  `json:"valid_to"`
	OrderCasemark             null.String `json:"order_casemark"`
	OrderDate                 *time.Time  `json:"order_date"`
	UploadOrderDocName        null.String `json:"upload_order_doc_name"`
	EstablishmentRegisterID   null.Int    `json:"establishment_register_id"`
	Designation               null.String `json:"designation"`
	PermanentStatus           null.String `json:"permanent_status"`
	EstablishmentRegisterName null.String `json:"establishment_register_name"`
	GroupName                 null.String `json:"group_name"`
	OfficeType                null.String `json:"office_type"`
	OfficeSupervisor          null.String `json:"office_supervisor"`
	IsHeadOfTheOffice         *bool       `json:"is_head_of_the_office"`
	PostID                    null.Int    `json:"post_id"` //post_id
	CadreName                 null.String `json:"cadre_name"`
	GroupID                   null.Int    `json:"group_id"`       //group_id
	CadreID                   null.Int    `json:"cadre_id"`       //cadre_id
	DesignationID             null.Int    `json:"designation_id"` //designation_id
	PostName                  null.String `json:"post_name"`      //post_name
	EmployeeName              null.String `json:"employee_name"`
}

func NewGetPostDetailsForRedeploymentResponse(posts []domain.PostDetailsForRedeployment) []GetPostDetailsForRedeploymentResponse {
	rsp := make([]GetPostDetailsForRedeploymentResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetPostDetailsForRedeploymentResponse{
			OfficeID:                  post.OfficeID,
			OfficeName:                post.OfficeName,
			FilledStatus:              post.FilledStatus,
			PostStatus:                post.PostStatus,
			AllowancesAttached:        post.AllowancesAttached,
			AllowanceDescription:      post.AllowanceDescription,
			UpdatedBy:                 post.UpdatedBy,
			UpdatedDate:               post.UpdatedDate,
			Status:                    post.Status,
			Remarks:                   post.Remarks,
			ValidFrom:                 post.ValidFrom,
			ValidTo:                   post.ValidTo,
			OrderCasemark:             post.OrderCasemark,
			OrderDate:                 post.OrderDate,
			UploadOrderDocName:        post.UploadOrderDocName,
			EstablishmentRegisterID:   post.EstablishmentRegisterID,
			Designation:               post.Designation,
			PermanentStatus:           post.PermanentStatus,
			EstablishmentRegisterName: post.EstablishmentRegisterName,
			GroupName:                 post.GroupName,
			OfficeType:                post.OfficeType,
			OfficeSupervisor:          post.OfficeSupervisor,
			IsHeadOfTheOffice:         post.IsHeadOfTheOffice,
			PostID:                    post.PostID,
			CadreName:                 post.CadreName,
			GroupID:                   post.GroupID,
			CadreID:                   post.CadreID,
			DesignationID:             post.DesignationID,
			PostName:                  post.PostName,
			EmployeeName:              post.EmployeeName,
		})
	}
	return rsp
}

type UpdatePostDetailsbyPostIDAPIResponse struct {
	StatusCodeAndMessage port.StatusCodeAndMessage `json:"status_code_and_message"`
	Data                 interface{}               `json:"data,omitempty"`
	Error                string                    `json:"error,omitempty"`
}

type GetPostDetailsAPIResponse struct {
	StatusCode int                      `json:"status_code"`
	Message    string                   `json:"message"`
	Data       []GetPostDetailsResponse `json:"data"`
	TotalCount int                      `json:"total"`
}

type GetPostDetailsResponse struct {
	PostID   null.Int    `json:"post_id"`
	PostName null.String `json:"post_name"`
}

func NewGetPostDetailsResponse(posts []domain.PostNameDetails) []GetPostDetailsResponse {
	rsp := make([]GetPostDetailsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetPostDetailsResponse{
			PostID:   post.PostID,
			PostName: post.PostName,
		})
	}
	return rsp
}

type GetDesignationDetailsAPIResponse struct {
	StatusCode int                             `json:"status_code"`
	Message    string                          `json:"message"`
	Data       []GetDesignationDetailsResponse `json:"data"`
	TotalCount int                             `json:"total"`
}

type GetDesignationDetailsResponse struct {
	DesignationID null.Int    `json:"designation_id"`
	Designation   null.String `json:"designation"`
}

func NewGetDesignaationDetailsResponse(posts []domain.DesignationDetails) []GetDesignationDetailsResponse {
	rsp := make([]GetDesignationDetailsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetDesignationDetailsResponse{
			DesignationID: post.DesignationID,
			Designation:   post.Designation,
		})
	}
	return rsp
}

func NewExceptionReportOrderCasemarkResponse(
	offices []domain.ExceptionReport,
) []domain.ExceptionReport {
	rsp := make([]domain.ExceptionReport, 0, len(offices))
	for _, office := range offices {
		rsp = append(rsp, domain.ExceptionReport{
			OfficeID:        office.OfficeID,
			GroupName:       office.GroupName,
			CadreName:       office.CadreName,
			PostID:          office.PostID,
			PostName:        office.PostName,
			CircleName:      office.CircleName,
			RegionName:      office.RegionName,
			DivisionName:    office.DivisionName,
			SubDivisionName: office.SubDivisionName,
			OfficeName:      office.OfficeName,
		})
	}
	return rsp
}

type ExceptionReportOrderCasemarkAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []domain.ExceptionReport `json:"data"`
}

type GetPostRedeployedInwardReportsAPIResponse struct {
	StatusCode int                                      `json:"status_code"`
	Message    string                                   `json:"message"`
	Data       []GetPostRedeployedInwardReportsResponse `json:"data"`
	TotalCount int                                      `json:"total"`
}

type GetPostRedeployedInwardReportsResponse struct {
	PostID                   int         `json:"post_id"`
	CadreName                string      `json:"cadre_name"`
	RedeploymentFromOfficeID int         `json:"redeployment_from_office_id"`
	RedeploymentToOfficeID   int         `json:"redeployment_to_office_id"`
	RedeploymentOn           time.Time   `json:"redeployment_on"`
	RedeploymentBy           string      `json:"redeployment_by"`
	EffectiveFrom            time.Time   `json:"effective_from"`
	EffectiveUpto            time.Time   `json:"effective_upto"`
	UploadOrderDocName       null.String `json:"upload_order_doc_name"`
}

func NewGetPostRedeployedInwardReportsResponse(posts []domain.PostRedeploymentReport) []GetPostRedeployedInwardReportsResponse {
	rsp := make([]GetPostRedeployedInwardReportsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetPostRedeployedInwardReportsResponse{
			PostID:                   post.PostID,
			CadreName:                post.CadreName,
			RedeploymentFromOfficeID: post.RedeploymentFromOfficeID,
			RedeploymentToOfficeID:   post.RedeploymentToOfficeID,
			RedeploymentOn:           post.RedeploymentOn,
			RedeploymentBy:           post.RedeploymentBy,
			EffectiveFrom:            post.EffectiveFrom,
			EffectiveUpto:            post.EffectiveUpto,
			UploadOrderDocName:       post.UploadOrderDocName,
		})
	}
	return rsp
}

type GetPostRedeployedOutwardReportsAPIResponse struct {
	StatusCode int                                       `json:"status_code"`
	Message    string                                    `json:"message"`
	Data       []GetPostRedeployedOutwardReportsResponse `json:"data"`
	TotalCount int                                       `json:"total"`
}

type GetPostRedeployedOutwardReportsResponse struct {
	PostID                   int         `json:"post_id"`
	CadreName                string      `json:"cadre_name"`
	RedeploymentFromOfficeID int         `json:"redeployment_from_office_id"`
	RedeploymentToOfficeID   int         `json:"redeployment_to_office_id"`
	RedeploymentOn           time.Time   `json:"redeployment_on"`
	RedeploymentBy           string      `json:"redeployment_by"`
	EffectiveFrom            time.Time   `json:"effective_from"`
	EffectiveUpto            time.Time   `json:"effective_upto"`
	UploadOrderDocName       null.String `json:"upload_order_doc_name"`
}

func NewGetPostRedeployedOutwardReportsResponse(posts []domain.PostRedeploymentReport) []GetPostRedeployedOutwardReportsResponse {
	rsp := make([]GetPostRedeployedOutwardReportsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetPostRedeployedOutwardReportsResponse{
			PostID:                   post.PostID,
			CadreName:                post.CadreName,
			RedeploymentFromOfficeID: post.RedeploymentFromOfficeID,
			RedeploymentToOfficeID:   post.RedeploymentToOfficeID,
			RedeploymentOn:           post.RedeploymentOn,
			RedeploymentBy:           post.RedeploymentBy,
			EffectiveFrom:            post.EffectiveFrom,
			EffectiveUpto:            post.EffectiveUpto,
			UploadOrderDocName:       post.UploadOrderDocName,
		})
	}
	return rsp
}

type GetCadreWiseReportsAPIResponse struct {
	StatusCode int                           `json:"status_code"`
	Message    string                        `json:"message"`
	Data       []GetCadreWiseReportsResponse `json:"data"`
	TotalCount int                           `json:"total"`
}

type GetCadreWiseReportsResponse struct {
	CadreID          null.Int `json:"cadre_id"`
	CadreName        string   `json:"cadre_name"`
	TotalPosts       int      `json:"total_posts"`
	TotalFilledPosts int      `json:"total_filled_posts"`
	TotalVacantPosts int      `json:"total_vacant_posts"`
}

func NewGetCadreWiseReportsResponse(posts []domain.CadreWiseReports) []GetCadreWiseReportsResponse {
	rsp := make([]GetCadreWiseReportsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetCadreWiseReportsResponse{
			CadreID:          post.CadreID,
			CadreName:        post.CadreName,
			TotalPosts:       post.TotalPosts,
			TotalFilledPosts: post.TotalFilledPosts,
			TotalVacantPosts: post.TotalVacantPosts,
		})
	}
	return rsp
}

type GetCadreWiseofficewiseReportsAPIResponse struct {
	StatusCode int                                     `json:"status_code"`
	Message    string                                  `json:"message"`
	Data       []GetCadreWiseofficewiseReportsResponse `json:"data"`
	TotalCount int                                     `json:"total"`
}

type GetCadreWiseofficewiseReportsResponse struct {
	GroupOfficeID    int    `json:"group_office_id"`
	GroupOfficeName  string `json:"group_office_name"`
	OfficeID         int    `json:"office_id"`
	OfficeName       string `json:"office_name"`
	CadreID          int    `json:"cadre_id"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

func NewGetCadreWiseOfficeWiseReportsResponse(posts []domain.CadreWiseOfficeWiseReport) []GetCadreWiseofficewiseReportsResponse {
	rsp := make([]GetCadreWiseofficewiseReportsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, GetCadreWiseofficewiseReportsResponse{
			GroupOfficeID:    post.GroupOfficeID,
			GroupOfficeName:  post.GroupOfficeName,
			OfficeID:         post.OfficeID,
			OfficeName:       post.OfficeName,
			CadreID:          post.CadreID,
			CadreName:        post.CadreName,
			TotalPosts:       post.TotalPosts,
			TotalFilledPosts: post.TotalFilledPosts,
			TotalVacantPosts: post.TotalVacantPosts,
		})
	}
	return rsp
}

type SavePostRedeployment2APIResponse struct {
	StatusCodeAndMessage port.StatusCodeAndMessage `json:"status_code_and_message"`
	FileName             string                    `json:"fileName"`
}
type GetRedeployedPostAuthorityChargesAPIResponse struct {
	StatusCode int                                         `json:"status_code"`
	Message    string                                      `json:"message"`
	Data       []GetRedeployedPostAuthorityChargesResponse `json:"data"`
	TotalCount int                                         `json:"total"`
}

type GetRedeployedPostAuthorityChargesResponse struct {
	AuthorityName   string                      `json:"authority_name"`
	EmployeeCount   int                         `json:"employee_count"`
	EmployeeDetails []GetEmployeeDetailResponse `json:"employee_details"`
}

type GetEmployeeDetailResponse struct {
	EmployeeName null.String `json:"employee_name"`
	EmployeeID   null.Int    `json:"employee_id"`
	PostID       null.Int    `json:"post_id"`
	PostName     null.String `json:"post_name"`
	OfficeID     null.Int    `json:"office_id"`
	OfficeName   null.String `json:"office_name"`
}

func NewGetRedeployedPostAuthorityChargesResponse(posts []domain.RedeployedPostAuthority) []GetRedeployedPostAuthorityChargesResponse {
	rsp := make([]GetRedeployedPostAuthorityChargesResponse, 0, len(posts))
	for _, post := range posts {
		employeeDetailsRsp := make([]GetEmployeeDetailResponse, 0, len(post.EmployeeDetails))
		for _, ed := range post.EmployeeDetails {
			employeeDetailsRsp = append(employeeDetailsRsp, GetEmployeeDetailResponse{
				EmployeeName: ed.EmployeeName,
				EmployeeID:   ed.EmployeeID,
				PostID:       ed.PostID,
				PostName:     ed.PostName,
				OfficeID:     ed.OfficeID,
				OfficeName:   ed.OfficeName,
			})
		}
		rsp = append(rsp, GetRedeployedPostAuthorityChargesResponse{
			AuthorityName:   post.AuthorityName,
			EmployeeCount:   post.EmployeeCount,
			EmployeeDetails: employeeDetailsRsp,
		})
	}
	return rsp
}

type GenericSuccessResponse struct {
	StatusCode int    `json:"statusCode"`
	Message    string `json:"message"`
}

type SummaryResponse struct {
	GroupName   string `json:"group_name"`
	CadreName   string `json:"cadre_name"`
	TotalPosts  int    `json:"total_posts"`
	TotalFilled int    `json:"total_filled_posts"`
	TotalVacant int    `json:"total_vacant_posts"`
}

type DetailResponse struct {
	CircleName     string `json:"circle_name"`
	CircleCode     string `json:"circle_code"`
	CircleOfficeID int64  `json:"circle_office_id"`
	OfficeName     string `json:"office_name"`
	OfficeID       int64  `json:"office_id"`
	OfficeTypeCode string `json:"office_type_code"`
	Pincode        string `json:"pincode"`
	DivisionName   string `json:"division_name"`
	PostID         int64  `json:"post_id"`
	PostName       string `json:"post_name"`
	Designation    string `json:"designation"`
	FilledStatus   string `json:"filled_status"`
	PostStatus     string `json:"post_status"`
}

type CadreInfoResponse struct {
	CadreID   int64  `json:"cadre_id"`
	CadreName string `json:"cadre_name"`
	GroupName string `json:"group_name"`
}

type CountSummaryResponse struct {
	TotalPosts  int `json:"total_posts"`
	TotalFilled int `json:"total_filled_posts"`
	TotalVacant int `json:"total_vacant_posts"`
}

func NewSummaryResponses(domainData []domain.Summary) []SummaryResponse {
	var res []SummaryResponse
	for _, d := range domainData {
		res = append(res, SummaryResponse{
			GroupName:   d.GroupName,
			CadreName:   d.CadreName,
			TotalPosts:  d.TotalPosts,
			TotalFilled: d.TotalFilled,
			TotalVacant: d.TotalVacant,
		})
	}
	return res
}

func NewDetailResponses(domainData []domain.Detail) []DetailResponse {
	var res []DetailResponse
	for _, d := range domainData {
		res = append(res, DetailResponse{
			CircleName:     d.CircleName,
			CircleCode:     d.CircleCode,
			CircleOfficeID: d.CircleOfficeID,
			OfficeName:     d.OfficeName,
			OfficeID:       d.OfficeID,
			OfficeTypeCode: d.OfficeTypeCode,
			Pincode:        d.Pincode,
			DivisionName:   d.DivisionName,
			PostID:         d.PostID,
			PostName:       d.PostName,
			Designation:    d.Designation,
			FilledStatus:   d.FilledStatus,
			PostStatus:     d.PostStatus,
		})
	}
	return res
}

func NewCadreInfoResponse(domainCadre domain.CadreInfo) CadreInfoResponse {
	return CadreInfoResponse{
		CadreID:   domainCadre.CadreID,
		CadreName: domainCadre.CadreName,
		GroupName: domainCadre.GroupName,
	}
}

type CadreReportResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []SummaryResponse    `json:"data"`
	Summary                   CountSummaryResponse `json:"summary"`
	CadreInfo                 CadreInfoResponse    `json:"cadreInfo"`
	DetailList                []DetailResponse     `json:"detailList,omitempty"`
}

func NewCountSummaryResponse(totalPosts, totalFilled, totalVacant int) CountSummaryResponse {
	return CountSummaryResponse{
		TotalPosts:  totalPosts,
		TotalFilled: totalFilled,
		TotalVacant: totalVacant,
	}
}

type CircleSummaryResponse struct {
	CircleName       string `json:"circle_name"`
	CircleOfficeID   int    `json:"circle_office_id"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

func NewCircleSummaryResponse(domianData []domain.CircleSummary) []CircleSummaryResponse {
	var res []CircleSummaryResponse
	for _, d := range domianData {
		res = append(res, CircleSummaryResponse{
			CircleName:       d.CircleName,
			CircleOfficeID:   d.CircleOfficeID,
			GroupName:        d.GroupName,
			CadreName:        d.CadreName,
			TotalPosts:       d.TotalPosts,
			TotalFilledPosts: d.TotalFilledPosts,
			TotalVacantPosts: d.TotalVacantPosts,
		})
	}
	return res
}

type DetailedPostResponse struct {
	PostManagementID   int    `json:"postmanagement_id"`
	PostID             string `json:"post_id"`
	PostName           string `json:"post_name"`
	Designation        string `json:"designation"`
	FilledStatus       string `json:"filled_status"`
	PostStatus         string `json:"post_status"`
	PayLevel           string `json:"pay_level"`
	GradePay           string `json:"grade_pay"`
	SanctionedStrength string `json:"sanctioned_strength"`
	PermanentStatus    bool   `json:"permanent_status"`
	AllowancesAttached bool   `json:"allowances_attached"`
	AllowanceDesc      string `json:"allowance_description"`
	GroupName          string `json:"group_name"`
	CadreName          string `json:"cadre_name"`
	PostOfficeName     string `json:"post_office_name"`
	OfficeID           int    `json:"office_id"`
	OfficeName         string `json:"office_name"`
	OfficeTypeCode     string `json:"office_type_code"`
	Pincode            string `json:"pincode"`
	DivisionName       string `json:"division_name"`
	SubdivisionName    string `json:"subdivision_name"`
	CircleName         string `json:"circle_name"`
	RegionName         string `json:"region_name"`
}

func NewDetailPostResponse(domainData []domain.DetailedPost) []DetailedPostResponse {
	var res []DetailedPostResponse
	for _, d := range domainData {
		res = append(res, DetailedPostResponse{
			PostManagementID:   d.PostManagementID,
			PostID:             d.PostID,
			PostName:           d.PostName,
			Designation:        d.Designation,
			FilledStatus:       d.FilledStatus,
			PostStatus:         d.PostStatus,
			PayLevel:           d.PayLevel,
			GradePay:           d.GradePay,
			SanctionedStrength: d.SanctionedStrength,
			PermanentStatus:    d.PermanentStatus,
			AllowancesAttached: d.AllowancesAttached,
			AllowanceDesc:      d.AllowanceDesc,
			GroupName:          d.GroupName,
			CadreName:          d.CadreName,
			PostOfficeName:     d.PostOfficeName,
			OfficeID:           d.OfficeID,
			OfficeName:         d.OfficeName,
			OfficeTypeCode:     d.OfficeTypeCode,
			Pincode:            d.Pincode,
			DivisionName:       d.DivisionName,
			SubdivisionName:    d.SubdivisionName,
			CircleName:         d.CircleName,
			RegionName:         d.RegionName,
		})
	}
	return res
}

type HierarchyInfoResponse struct {
	Level     int64  `json:"level"`
	LevelName string `json:"level_name"`
	CadreName string `json:"cadre_name"`
}

func NewHierarchyInfoResponse(domainData domain.HierarchyInfo) HierarchyInfoResponse {
	return HierarchyInfoResponse{
		Level:     domainData.Level,
		LevelName: domainData.LevelName,
		CadreName: domainData.CadreName,
	}
}

type CircleCadreReportResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []CircleSummaryResponse `json:"data"`
	Summary                   CountSummaryResponse    `json:"summary"`
	HierarchyInfo             HierarchyInfoResponse   `json:"heirarcht_info"`
	DetailList                []DetailedPostResponse  `json:"detailList,omitempty"`
}

type DivisionSummaryResponse struct {
	DivisionName     string `json:"division_name"`
	DivisionOfficeID int    `json:"division_office_id"`
	RegionName       string `json:"region_name"`
	RegionOfficeID   int    `json:"region_office_id"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

func NewDivisionSummaryResponse(domainData []domain.DivisionSummary) []DivisionSummaryResponse {
	var res []DivisionSummaryResponse
	for _, d := range domainData {
		res = append(res, DivisionSummaryResponse{
			DivisionName:     d.DivisionName,
			DivisionOfficeID: d.DivisionOfficeID,
			RegionName:       d.RegionName,
			RegionOfficeID:   d.RegionOfficeID,
			GroupName:        d.GroupName,
			CadreName:        d.CadreName,
			TotalPosts:       d.TotalPosts,
			TotalFilledPosts: d.TotalFilledPosts,
			TotalVacantPosts: d.TotalVacantPosts,
		})
	}
	return res
}

type DivisionDetailResponse struct {
	PostManagementID   int    `json:"postmanagement_id"`
	PostID             string `json:"post_id"`
	PostName           string `json:"post_name"`
	Designation        string `json:"designation"`
	FilledStatus       string `json:"filled_status"`
	PostStatus         string `json:"post_status"`
	PayLevel           string `json:"pay_level"`
	GradePay           string `json:"grade_pay"`
	SanctionedStrength string `json:"sanctioned_strength"`
	PermanentStatus    bool   `json:"permanent_status"`
	AllowancesAttached bool   `json:"allowances_attached"`
	AllowanceDesc      string `json:"allowance_description"`
	GroupName          string `json:"group_name"`
	CadreName          string `json:"cadre_name"`
	PostOfficeName     string `json:"post_office_name"`
	OfficeID           int    `json:"office_id"`
	OfficeName         string `json:"office_name"`
	OfficeTypeCode     string `json:"office_type_code"`
	Pincode            string `json:"pincode"`
	DivisionName       string `json:"division_name"`
	SubdivisionName    string `json:"subdivision_name"`
	CircleName         string `json:"circle_name"`
	RegionName         string `json:"region_name"`
}

func NewDetailListResponse(domainData []domain.DivisionDetail) []DivisionDetailResponse {
	var res []DivisionDetailResponse
	for _, d := range domainData {
		res = append(res, DivisionDetailResponse{
			PostManagementID:   d.PostManagementID,
			PostID:             d.PostID,
			PostName:           d.PostName,
			Designation:        d.Designation,
			FilledStatus:       d.FilledStatus,
			PostStatus:         d.PostStatus,
			PayLevel:           d.PayLevel,
			GradePay:           d.GradePay,
			SanctionedStrength: d.SanctionedStrength,
			PermanentStatus:    d.PermanentStatus,
			AllowancesAttached: d.AllowancesAttached,
			AllowanceDesc:      d.AllowanceDesc,
			GroupName:          d.GroupName,
			CadreName:          d.CadreName,
			PostOfficeName:     d.PostOfficeName,
			OfficeID:           d.OfficeID,
			OfficeName:         d.OfficeName,
			OfficeTypeCode:     d.OfficeTypeCode,
			Pincode:            d.Pincode,
			DivisionName:       d.DivisionName,
			SubdivisionName:    d.SubdivisionName,
			CircleName:         d.CadreName,
			RegionName:         d.RegionName,
		})
	}
	return res
}

type RegionInfoResponse struct {
	Level          int64  `json:"level"`
	LevelName      string `json:"level_name"`
	CadreName      string `json:"cadre_name"`
	RegionOfficeId int64  `json:"region_office_id"`
	RegionName     string `json:"region_name"`
	CircleName     string `json:"circle_name"`
	CircleOfficeID int    `json:"circle_office_id"`
}

func NewRegionInfoResponse(domaindata domain.RegionInfo) RegionInfoResponse {
	return RegionInfoResponse{
		Level:          domaindata.Level,
		LevelName:      domaindata.LevelName,
		CadreName:      domaindata.CadreName,
		RegionOfficeId: domaindata.RegionOfficeId,
		RegionName:     domaindata.RegionName,
		CircleName:     domaindata.CircleName,
		CircleOfficeID: domaindata.CircleOfficeID,
	}
}

type DivisionCadreReportResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []DivisionSummaryResponse `json:"data"`
	Summary                   CountSummaryResponse      `json:"summary"`
	HierarchyInfo             RegionInfoResponse        `json:"hierarchy_info"`
	DetailList                []DivisionDetailResponse  `json:"detailList,omitempty"`
}

type HierarchySummaryResponse struct {
	OfficeID          int64  `json:"office_id"`
	OfficeName        string `json:"office_name"`
	OfficeTypeCode    string `json:"office_type_code"`
	Pincode           string `json:"pincode"`
	ReportingOfficeID int64  `json:"reporting_office_id"`
	DivisionName      string `json:"division_name"`
	SubdivisionName   string `json:"subdivision_name"`
	CircleName        string `json:"circle_name"`
	TotalPosts        int    `json:"total_posts"`
	TotalFilledPosts  int    `json:"total_filled_posts"`
	TotalVacantPosts  int    `json:"total_vacant_posts"`
}

func NewHierarchySummaryResponse(domainData []domain.HierarchySummary) []HierarchySummaryResponse {
	var res []HierarchySummaryResponse
	for _, d := range domainData {
		res = append(res, HierarchySummaryResponse{
			OfficeID:          d.OfficeID,
			OfficeName:        d.OfficeName,
			OfficeTypeCode:    d.OfficeTypeCode,
			Pincode:           d.Pincode,
			ReportingOfficeID: d.ReportingOfficeID,
			DivisionName:      d.DivisionName,
			SubdivisionName:   d.SubdivisionName,
			CircleName:        d.CircleName,
			TotalPosts:        d.TotalPosts,
			TotalFilledPosts:  d.TotalFilledPosts,
			TotalVacantPosts:  d.TotalVacantPosts,
		})
	}
	return res
}

type HierarchyDetailResponse struct {
	ParentOfficeName string `json:"parent_office_name"`
	ParentOfficeID   int64  `json:"parent_office_id"`
	PostManagementID int64  `json:"postmanagement_id"`
	PostID           string `json:"post_id"`
	PostName         string `json:"post_name"`
	Designation      string `json:"designation"`
	FilledStatus     string `json:"filled_status"`
	PostStatus       string `json:"post_status"`
	PostOfficeName   string `json:"post_office_name"`
	PostOfficeID     int64  `json:"post_office_id"`
}

func NewHierarchyDetailResponse(domainData []domain.HierarchyDetail) []HierarchyDetailResponse {
	var res []HierarchyDetailResponse
	for _, d := range domainData {
		res = append(res, HierarchyDetailResponse{
			ParentOfficeName: d.ParentOfficeName,
			ParentOfficeID:   d.ParentOfficeID,
			PostManagementID: d.PostManagementID,
			PostID:           d.PostID,
			PostName:         d.PostName,
			Designation:      d.Designation,
			FilledStatus:     d.FilledStatus,
			PostStatus:       d.PostStatus,
			PostOfficeName:   d.PostOfficeName,
			PostOfficeID:     d.PostOfficeID,
		})
	}
	return res
}

type HierarchyInfodataResponse struct {
	ParentOfficeID   int64  `json:"parent_office_id"`
	ParentOfficeName string `json:"parent_office_name"`
	CadreName        string `json:"cadre_name"`
	Level            int    `json:"level"`
	OfficeTypeCode   string `json:"office_type_code"`
}

func NewHierarchyInfodataResponse(domaindata domain.HierarchyInfodata) HierarchyInfodataResponse {
	return HierarchyInfodataResponse{
		ParentOfficeID:   domaindata.ParentOfficeID,
		ParentOfficeName: domaindata.ParentOfficeName,
		CadreName:        domaindata.CadreName,
		Level:            domaindata.Level,
	}
}

type HierarchyReportResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []HierarchySummaryResponse `json:"data"`
	Summary                   CountSummaryResponse       `json:"summary"`
	HierarchyInfo             HierarchyInfodataResponse  `json:"hierarchy_info"`
	DetailList                []HierarchyDetailResponse  `json:"detailList,omitempty"`
}

type OfficeDataResponse struct {
	OfficeID         int64  `json:"office_id"`
	OfficeName       string `json:"office_name"`
	OfficeTypeCode   string `json:"office_type_code"`
	Pincode          string `json:"pincode"`
	EmailID          string `json:"email_id"`
	ContactNumber    string `json:"contact_number"`
	DivisionName     string `json:"division_name"`
	DivisionOfficeID int64  `json:"division_office_id"`
	SubdivisionName  string `json:"subdivision_name"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

func NewOfficeSummaryResponse(domainData []domain.OfficeData) []OfficeDataResponse {
	var res []OfficeDataResponse
	for _, d := range domainData {
		res = append(res, OfficeDataResponse{
			OfficeID:         d.OfficeID,
			OfficeName:       d.OfficeName,
			OfficeTypeCode:   d.OfficeTypeCode,
			Pincode:          d.Pincode,
			EmailID:          d.EmailID,
			ContactNumber:    d.ContactNumber,
			DivisionName:     d.DivisionName,
			DivisionOfficeID: d.DivisionOfficeID,
			SubdivisionName:  d.SubdivisionName,
			GroupName:        d.GroupName,
			CadreName:        d.CadreName,
			TotalPosts:       d.TotalPosts,
			TotalFilledPosts: d.TotalFilledPosts,
			TotalVacantPosts: d.TotalVacantPosts,
		})
	}
	return res
}

type OfficePostDetailResponse struct {
	PostManagementID     int64  `json:"postmanagement_id"`
	PostID               string `json:"post_id"`
	PostName             string `json:"post_name"`
	Designation          string `json:"designation"`
	FilledStatus         string `json:"filled_status"`
	PostStatus           string `json:"post_status"`
	PayLevel             string `json:"pay_level"`
	GradePay             string `json:"grade_pay"`
	SanctionedStrength   string `json:"sanctioned_strength"`
	PermanentStatus      string `json:"permanent_status"`
	AllowancesAttached   string `json:"allowances_attached"`
	AllowanceDescription string `json:"allowance_description"`
	GroupName            string `json:"group_name"`
	CadreName            string `json:"cadre_name"`
	PostOfficeName       string `json:"post_office_name"`
	OfficeID             int64  `json:"office_id"`
	OfficeName           string `json:"office_name"`
	OfficeTypeCode       string `json:"office_type_code"`
	Pincode              string `json:"pincode"`
	DivisionName         string `json:"division_name"`
	SubdivisionName      string `json:"subdivision_name"`
	CircleName           string `json:"circle_name"`
	RegionName           string `json:"region_name"`
}

func NewOfficeDetailResponse(domainData []domain.OfficePostDetail) []OfficePostDetailResponse {
	var res []OfficePostDetailResponse
	for _, d := range domainData {
		res = append(res, OfficePostDetailResponse{
			PostManagementID:     d.PostManagementID,
			PostID:               d.PostID,
			PostName:             d.PostName,
			Designation:          d.Designation,
			FilledStatus:         d.FilledStatus,
			PostStatus:           d.PostStatus,
			PayLevel:             d.PayLevel,
			GradePay:             d.GradePay,
			SanctionedStrength:   d.SanctionedStrength,
			PermanentStatus:      d.PermanentStatus,
			AllowancesAttached:   d.AllowancesAttached,
			AllowanceDescription: d.AllowanceDescription,
			GroupName:            d.GroupName,
			CadreName:            d.CadreName,
			PostOfficeName:       d.PostOfficeName,
			OfficeID:             d.OfficeID,
			OfficeName:           d.OfficeName,
			OfficeTypeCode:       d.OfficeTypeCode,
			Pincode:              d.Pincode,
			DivisionName:         d.DivisionName,
			SubdivisionName:      d.SubdivisionName,
			CircleName:           d.CircleName,
			RegionName:           d.RegionName,
		})
	}
	return res
}

type OfficeInfoResponse struct {
	Level            int    `json:"level"`
	LevelName        string `json:"level_name"`
	CadreName        string `json:"cadre_name"`
	DivisionOfficeId int64  `json:"division_office_id"`
	DivisionName     string `json:"division_name"`
	RegionName       string `json:"region_name"`
	RegionOfficeID   int64  `json:"region_office_id"`
	CircleName       string `json:"circle_name"`
	CircleOfficeID   int64  `json:"circle_office_id"`
}

func NewOfficeInfodataResponse(domainData domain.OfficeInfo) OfficeInfoResponse {
	return OfficeInfoResponse{
		Level:            domainData.Level,
		LevelName:        domainData.LevelName,
		CadreName:        domainData.CadreName,
		DivisionOfficeId: domainData.DivisionOfficeId,
		DivisionName:     domainData.DivisionName,
		RegionName:       domainData.RegionName,
		RegionOfficeID:   domainData.RegionOfficeID,
		CircleName:       domainData.CircleName,
		CircleOfficeID:   domainData.CircleOfficeID,
	}
}

type OfficeReportResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []OfficeDataResponse       `json:"data"`
	Summary                   CountSummaryResponse       `json:"summary"`
	HierarchyInfo             OfficeInfoResponse         `json:"hierarchy_info"`
	DetailList                []OfficePostDetailResponse `json:"detailList,omitempty"`
}

type HierarchyPostResponse struct {
	Level            int    `json:"level"`
	LevelName        string `json:"level_name"`
	CadreID          int64  `json:"cadre_id"`
	OfficeID         *int64 `json:"office_id,omitempty"`
	DivisionOfficeID *int64 `json:"division_office_id,omitempty"`
	RegionOfficeID   *int64 `json:"region_office_id,omitempty"`
	CircleOfficeID   *int64 `json:"circle_office_id,omitempty"`
}

func NewHierarchyPost(domainData domain.HierarchyPost) HierarchyPostResponse {
	return HierarchyPostResponse{
		Level:            domainData.Level,
		LevelName:        domainData.LevelName,
		CadreID:          domainData.CadreID,
		OfficeID:         domainData.OfficeID,
		DivisionOfficeID: domainData.DivisionOfficeID,
		RegionOfficeID:   domainData.RegionOfficeID,
		CircleOfficeID:   domainData.CircleOfficeID,
	}
}

type PostDetailResponse struct {
	PostManagementID     int64  `json:"postmanagement_id"`
	PostName             string `json:"post_name"`
	Designation          string `json:"designation"`
	FilledStatus         string `json:"filled_status"`
	PostStatus           string `json:"post_status"`
	PayLevel             string `json:"pay_level"`
	GradePay             string `json:"grade_pay"`
	SanctionedStrength   string `json:"sanctioned_strength"`
	PermanentStatus      string `json:"permanent_status"`
	AllowancesAttached   string `json:"allowances_attached"`
	AllowanceDescription string `json:"allowance_description"`
	GroupName            string `json:"group_name"`
	CadreName            string `json:"cadre_name"`
	PostOfficeName       string `json:"post_office_name"`
	OfficeID             int64  `json:"office_id"`
	OfficeName           string `json:"office_name"`
	OfficeTypeCode       string `json:"office_type_code"`
	Pincode              string `json:"pincode"`
	DivisionName         string `json:"division_name"`
	SubdivisionName      string `json:"subdivision_name"`
	CircleName           string `json:"circle_name"`
	RegionName           string `json:"region_name"`
}

func NewPostSummaryResponse(domainData []domain.PostDetail) []PostDetailResponse {
	var res []PostDetailResponse
	for _, d := range domainData {
		res = append(res, PostDetailResponse{
			PostManagementID:     d.PostManagementID,
			PostName:             d.PostName,
			Designation:          d.Designation,
			FilledStatus:         d.FilledStatus,
			PostStatus:           d.PostStatus,
			PayLevel:             d.PayLevel,
			GradePay:             d.GradePay,
			SanctionedStrength:   d.SanctionedStrength,
			PermanentStatus:      d.PermanentStatus,
			AllowancesAttached:   d.AllowancesAttached,
			AllowanceDescription: d.AllowanceDescription,
			GroupName:            d.GroupName,
			CadreName:            d.CadreName,
			PostOfficeName:       d.PostOfficeName,
			OfficeID:             d.OfficeID,
			OfficeName:           d.OfficeName,
			OfficeTypeCode:       d.OfficeTypeCode,
			Pincode:              d.Pincode,
			DivisionName:         d.DivisionName,
			SubdivisionName:      d.SubdivisionName,
			CircleName:           d.CadreName,
			RegionName:           d.RegionName,
		})
	}
	return res
}

type ContextInfoResponse struct {
	OfficeName     string `json:"office_name"`
	OfficeTypeCode string `json:"office_type_code"`
	DivisionName   string `json:"division_name"`
	RegionName     string `json:"region_name"`
	CircleName     string `json:"circle_name"`
}

func NewContextInfoResponse(domainData domain.ContextInfo) ContextInfoResponse {
	return ContextInfoResponse{
		OfficeName:     domainData.OfficeName,
		OfficeTypeCode: domainData.OfficeTypeCode,
		DivisionName:   domainData.DivisionName,
		RegionName:     domainData.RegionName,
		CircleName:     domainData.CircleName,
	}
}

type PostInfo struct {
	Level            int    `json:"level"`
	LevelName        string `json:"level_name"`
	CadreID          int64  `json:"cadre_id"`
	OfficeID         *int64 `json:"office_id,omitempty"`
	DivisionOfficeID *int64 `json:"division_office_id,omitempty"`
	RegionOfficeID   *int64 `json:"region_office_id,omitempty"`
	CircleOfficeID   *int64 `json:"circle_office_id,omitempty"`
}

type PostReport struct {
	CadreID          int64  `form:"cadre_id" binding:"required"`
	OfficeID         *int64 `form:"office_id"`
	CircleOfficeID   *int64 `form:"circle_office_id"`
	RegionOfficeID   *int64 `form:"region_office_id"`
	DivisionOfficeID *int64 `form:"division_office_id"`
	Search           string `form:"search"`
	IncludeList      string `form:"includeList"`
}

type PostSummaryResponse struct {
	TotalPosts  int `json:"totalPosts"`
	TotalFilled int `json:"totalFilled"`
	TotalVacant int `json:"totalVacant"`
}

func NewPostSummaryResponse1(domainData domain.PostSummary) PostSummaryResponse {
	return PostSummaryResponse{
		TotalPosts:  domainData.TotalPosts,
		TotalFilled: domainData.TotalFilled,
		TotalVacant: domainData.TotalVacant,
	}
}

type GetPostsResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	DetailList                []PostDetailResponse  `json:"detailList"`
	Summary                   PostSummaryResponse   `json:"summary"`
	ContextInfo               ContextInfoResponse   `json:"contextInfo"`
	HierarchyInfo             HierarchyPostResponse `json:"hierarchyInfo"`
}

type RegionSummaryResponse struct {
	RegionName       string `json:"region_name"`
	RegionOfficeID   int64  `json:"region_office_id"`
	CircleName       string `json:"circle_name"`
	CircleOfficeID   int64  `json:"circle_office_id"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

func NewRegionSummaryResponse(domainData []domain.RegionSummary) []RegionSummaryResponse {
	var res []RegionSummaryResponse
	for _, d := range domainData {
		res = append(res, RegionSummaryResponse{
			RegionName:       d.RegionName,
			RegionOfficeID:   d.RegionOfficeID,
			CircleName:       d.CircleName,
			CircleOfficeID:   d.CircleOfficeID,
			GroupName:        d.GroupName,
			CadreName:        d.CadreName,
			TotalPosts:       d.TotalPosts,
			TotalFilledPosts: d.TotalFilledPosts,
			TotalVacantPosts: d.TotalVacantPosts,
		})
	}
	return res
}

type RegionDetailResponse struct {
	PostManagementID   int64  `json:"postmanagement_id"`
	PostName           string `json:"post_name"`
	Designation        string `json:"designation"`
	FilledStatus       string `json:"filled_status"`
	PostStatus         string `json:"post_status"`
	PayLevel           string `json:"pay_level"`
	GradePay           string `json:"grade_pay"`
	SanctionedStrength string `json:"sanctioned_strength"`
	PermanentStatus    string `json:"permanent_status"`
	AllowancesAttached string `json:"allowances_attached"`
	AllowanceDesc      string `json:"allowance_description"`
	GroupName          string `json:"group_name"`
	CadreName          string `json:"cadre_name"`
	PostOfficeName     string `json:"post_office_name"`
	OfficeID           int64  `json:"office_id"`
	OfficeName         string `json:"office_name"`
	OfficeTypeCode     string `json:"office_type_code"`
	Pincode            string `json:"pincode"`
	DivisionName       string `json:"division_name"`
	SubdivisionName    string `json:"subdivision_name"`
	CircleName         string `json:"circle_name"`
	RegionName         string `json:"region_name"`
}

func NewRegionDetailResponse(domainData []domain.RegionDetail) []RegionDetailResponse {
	var res []RegionDetailResponse
	for _, d := range domainData {
		res = append(res, RegionDetailResponse{
			PostManagementID:   d.PostManagementID,
			PostName:           d.PostName,
			Designation:        d.Designation,
			FilledStatus:       d.FilledStatus,
			PostStatus:         d.PostStatus,
			PayLevel:           d.PayLevel,
			GradePay:           d.GradePay,
			SanctionedStrength: d.SanctionedStrength,
			PermanentStatus:    d.PermanentStatus,
			AllowancesAttached: d.AllowancesAttached,
			AllowanceDesc:      d.AllowanceDesc,
			GroupName:          d.GroupName,
			CadreName:          d.CadreName,
			PostOfficeName:     d.PostOfficeName,
			OfficeID:           d.OfficeID,
			OfficeName:         d.OfficeName,
			OfficeTypeCode:     d.OfficeTypeCode,
			Pincode:            d.Pincode,
			DivisionName:       d.DivisionName,
			SubdivisionName:    d.SubdivisionName,
			CircleName:         d.CircleName,
			RegionName:         d.RegionName,
		})
	}
	return res
}

type HierarchyRegionResponse struct {
	Level          int    `json:"level"`
	LevelName      string `json:"level_name"`
	CadreName      string `json:"cadre_name"`
	CircleOfficeID int64  `json:"circle_office_id"`
	CircleName     string `json:"circle_name"`
}

func NewHierarchyRegionResponse(domainData domain.HierarchyRegion) HierarchyRegionResponse {
	return HierarchyRegionResponse{
		Level:          domainData.Level,
		LevelName:      domainData.LevelName,
		CadreName:      domainData.CadreName,
		CircleOfficeID: domainData.CircleOfficeID,
		CircleName:     domainData.CircleName,
	}
}

func NewRegionCountResponse(domainData []domain.RegionSummary) RegionSummaryResponse {
	var total RegionSummaryResponse

	for _, item := range domainData {
		total.TotalPosts += item.TotalPosts
		total.TotalFilledPosts += item.TotalFilledPosts
		total.TotalVacantPosts += item.TotalVacantPosts
	}

	return total
}

type RegionReportResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []RegionSummaryResponse `json:"data"`
	Summary                   RegionSummaryResponse   `json:"summary"`
	HierarchyInfo             HierarchyRegionResponse `json:"hierarchy_info"`
	DetailList                []RegionDetailResponse  `json:"detailList,omitempty"`
}

type ListCadreWiseReportResponse struct {
	PostID        int    `json:"post_id" db:"post_id"`
	PostName      string `json:"post_name" db:"post_name"`
	OfficeId      int    `json:"office_id" db:"office_id"`
	OfficeName    string `json:"office_name" db:"office_name"`
	CadreId       int    `json:"cadre_id" db:"cadre_id"`
	CadreName     string `json:"cadre_name" db:"cadre_name"`
	DesignationID int    `json:"designation_id" db:"designation_id"`
	Designation   string `json:"designation" db:"designation"`
	FilledStatus  string `json:"filled_status" db:"post_status"`
}

func NewListCadreWiseOfficeWiseReportsResponse(posts []domain.ListCadreWiseReport) []ListCadreWiseReportResponse {
	rsp := make([]ListCadreWiseReportResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, ListCadreWiseReportResponse{
			PostID:        post.PostID,
			PostName:      post.PostName,
			OfficeId:      post.OfficeId,
			OfficeName:    post.OfficeName,
			CadreId:       post.CadreId,
			CadreName:     post.CadreName,
			DesignationID: post.DesignationID,
			Designation:   post.Designation,
			FilledStatus:  post.FilledStatus,
		})
	}
	return rsp
}

type ListCadreWiseOfficeAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      []ListCadreWiseReportResponse `json:"data"`
}

type PostAuthorityDetailsResponse struct {
	PostID               null.Int    `json:"post_id"`
	PostName             null.String `json:"post_name"`
	Designation          null.String `json:"designation"`
	CadreID              null.Int    `json:"cadre_id"`
	CadreName            null.String `json:"cadre_name"`
	GroupID              null.Int    `json:"group_id"`
	GroupName            null.String `json:"group_name"`
	EmployeeID           null.Int    `json:"employee_id"`
	EmployeeName         null.String `json:"employee_name"`
	CLSanctionAuthority  null.Int    `json:"cl_sanc_authority"`
	ELSanctionAuthority  null.Int    `json:"el_sanc_authority"`
	PayApproveAuthority1 null.Int    `json:"pay_approve_authority"`
	AppointingAuthority  null.Int    `json:"appointing_authority"`
	DisciplineAuthority  null.Int    `json:"discipline_authority"`
	DDOAuthority         null.Int    `json:"ddo_authority"`
}

type PostAuthorityDetailsAPIResponse struct {
	StatusCode int                            `json:"status_code"`
	Message    string                         `json:"message"`
	Data       []PostAuthorityDetailsResponse `json:"data"`
	TotalCount int                            `json:"total_records"`
}

func NewPostAuthorityDetailsResponse(posts []domain.PostAuthorityDetails) []PostAuthorityDetailsResponse {
	rsp := make([]PostAuthorityDetailsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, PostAuthorityDetailsResponse{
			PostID:               post.PostID,
			PostName:             post.PostName,
			Designation:          post.Designation,
			CadreID:              post.CadreID,
			CadreName:            post.CadreName,
			GroupID:              post.GroupID,
			GroupName:            post.GroupName,
			EmployeeID:           post.EmployeeID,
			EmployeeName:         post.EmployeeName,
			CLSanctionAuthority:  post.CLSanctionAuthority,
			ELSanctionAuthority:  post.ELSanctionAuthority,
			PayApproveAuthority1: post.PayApproveAuthority1,
			AppointingAuthority:  post.AppointingAuthority,
			DisciplineAuthority:  post.DisciplineAuthority,
			DDOAuthority:         post.DDOAuthority,
		})
	}
	return rsp
}

type DeletePostsbyOfficeIDAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      DeletePostsbyOfficeIDResponse `json:"data"`
}

type DeletePostsbyOfficeIDResponse struct {
	OfficeID int    `json:"office_id"`
	Message  string `json:"message"`
	Deleted  int    `json:"deleted"`
}

type CheckCadresAPIResponse struct {
	Data CheckCadreData `json:"data"`
}

type CheckCadreData struct {
	CadreExists bool `json:"cadre_exists"`
}

type SanctionedStrengthDetailsResponse struct {
	GroupID            int    `json:"group_id" db:"group_id"`
	CadreID            int    `json:"cadre_id" db:"cadre_id"`
	CadreName          string `json:"cadre_name" db:"cadre_name"`
	OfficeID           int    `json:"office_id" db:"office_id"`
	OfficeName         string `json:"office_name" db:"office_name"`
	SanctionedStrength int    `json:"sanctioned_strength"`
}

func NewSanctionedStrengthResponse(posts []domain.SanctionedStrengthDetails) []SanctionedStrengthDetailsResponse {
	rsp := make([]SanctionedStrengthDetailsResponse, 0, len(posts))
	for _, post := range posts {
		rsp = append(rsp, SanctionedStrengthDetailsResponse{
			GroupID:            post.GroupID,
			CadreID:            post.CadreID,
			CadreName:          post.CadreName,
			OfficeID:           post.OfficeID,
			OfficeName:         post.OfficeName,
			SanctionedStrength: post.SanctionedStrength,
		})
	}
	return rsp
}

type ListSanctionedStrengthAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []SanctionedStrengthDetailsResponse `json:"data"`
}

type UpdatePostnameByIdResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      string `json:"data"`
}

type PostdetailsResponse struct {
	PostManagementID  int64     `json:"postmanagement_id" db:"postmanagement_id"`
	PostID            int64     `json:"post_id" db:"post_id"`
	OfficeID          int64     `json:"office_id" db:"office_id"`
	OfficeName        string    `json:"office_name" db:"office_name"`
	PostName          string    `json:"post_name" db:"post_name"`
	GroupID           int64     `json:"group_id" db:"group_id"`
	CadreName         string    `json:"cadre_name" db:"cadre_name"`
	FilledStatus      string    `json:"filled_status" db:"filled_status"`
	Status            string    `json:"status" db:"status"`
	Remarks           string    `json:"remarks" db:"remarks"`
	ValidFrom         time.Time `json:"valid_from" db:"valid_from"`
	ValidTo           time.Time `json:"valid_to" db:"valid_to"`
	OrderCasemark     string    `json:"order_casemark" db:"order_casemark"`
	OrderDate         time.Time `json:"order_date" db:"order_date"`
	Designation       string    `json:"designation" db:"designation"`
	PayLevel          string    `json:"pay_level" db:"pay_level"`
	GradePay          string    `json:"grade_pay" db:"grade_pay"`
	PermanentStatus   bool      `json:"permanent_status" db:"permanent_status"`
	GroupName         string    `json:"group_name" db:"group_name"`
	CadreID           int64     `json:"cadre_id" db:"cadre_id"`
	DesignationID     int64     `json:"designation_id" db:"designation_id"`
	IsHeadOfTheOffice bool      `json:"is_head_of_the_office" db:"is_head_of_the_office"`
}

func NewGetPostDetailsbyPostIDAPIResponse(posts domain.Postdetails) PostdetailsResponse {
	return PostdetailsResponse{
		PostManagementID:  posts.PostManagementID,
		PostID:            posts.PostID,
		OfficeID:          posts.OfficeID,
		OfficeName:        posts.OfficeName,
		PostName:          posts.PostName,
		GroupID:           posts.GroupID,
		CadreName:         posts.CadreName,
		FilledStatus:      posts.FilledStatus,
		Status:            posts.Status,
		Remarks:           posts.Remarks,
		ValidFrom:         posts.ValidFrom,
		ValidTo:           posts.ValidTo,
		OrderCasemark:     posts.OrderCasemark,
		OrderDate:         posts.OrderDate,
		Designation:       posts.Designation,
		PayLevel:          posts.PayLevel,
		GradePay:          posts.GradePay,
		PermanentStatus:   posts.PermanentStatus,
		GroupName:         posts.GroupName,
		CadreID:           posts.CadreID,
		DesignationID:     posts.DesignationID,
		IsHeadOfTheOffice: posts.IsHeadOfTheOffice,
	}
}

type GetPostDetailsbyPostIDAPIResponse1 struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      PostdetailsResponse `json:"data"`
}

type CaddreSummaryResponse struct {
	Summary []domain.PostDetail1
	List    []domain.PostSummaryDetail
	Total   int
}

type GetPostManagementSummaryAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      CaddreSummaryResponse `json:"data"`
}

type CircleSummaryResponse1 struct {
	Summary   []domain.CircleSummaryDetail
	List      []domain.PostSummaryDetail
	Hierarchy domain.CircleHierarchy
	Total     int
}

type GetPostManagementCircleSummaryAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      CircleSummaryResponse1 `json:"data"`
}

type RegionSummaryResponse1 struct {
	Summary   []domain.RegionSummaryDetail
	List      []domain.PostSummaryDetail
	Hierarchy domain.RegionHierarchy
	Total     int
}

type GetPostManagementRegionSummaryAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      RegionSummaryResponse1 `json:"data"`
}

type DivisionSummaryResponse1 struct {
	Summary   []domain.DivisionSummaryDetail
	List      []domain.PostSummaryDetail
	Hierarchy domain.DivisionHierarchyInfo
	Total     int
}

type GetPostManagementDivisionSummaryAPIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      DivisionSummaryResponse1 `json:"data"`
}

type PostSummaryResponse1 struct {
	Summary []domain.PostSummaryDetail1
	List    []domain.PostSummaryDetail
	Total   int
}

type GetPostManagementDivisionSummaryAPIResponse1 struct {
	port.StatusCodeAndMessage `json:",inline"`
	Data                      PostSummaryResponse1 `json:"data"`
}

func NewFetchPostsByOfficeIDAllResponse2(
	masters []domain.PostManagementMasterNew,
) []FetchPostsByOfficeIDResponseNew {
	var response []FetchPostsByOfficeIDResponseNew
	for _, item := range masters {
		resp := FetchPostsByOfficeIDResponseNew{
			OfficeID:                  item.OfficeID,
			PostID:                    item.PostID,
			PostName:                  item.PostName,
			OfficeName:                item.OfficeName,
			GroupId:                   item.GroupId,
			GroupName:                 item.GroupName,
			CadreID:                   item.CadreID,
			CadreName:                 item.CadreName,
			DesignationID:             item.DesignationId,
			Designation:               item.Designation,
			FilledStatus:              item.FilledStatus,
			PostStatus:                item.PostStatus,
			Status:                    item.Status,
			PermanentStatus:           item.PermanentStatus,
			AllowancesAttached:        item.AllowancesAttached,
			AllowanceDescription:      item.AllowanceDescription,
			CreatedBy:                 item.CreatedBy,
			CreatedOn:                 item.CreatedDate,
			ApprovedBy:                item.ApprovedBy,
			ApprovedOn:                item.ApprovedDate,
			UpdatedBy:                 item.UpdatedBy,
			UpdatedOn:                 item.UpdatedDate,
			ValidFrom:                 item.ValidFrom,
			ValidTo:                   item.ValidTo,
			OrderCaseMark:             item.OrderCaseMark,
			OrderDate:                 item.OrderDate,
			UploadOrderDocName:        item.UploadOrderDocName,
			EstablishmentRegisterID:   item.EstablishmentRegisterID,
			EstablishmentRegisterName: item.EstablishmentRegisterName,
			EmployeeGroup:             item.EmployeeGroup,
			SanctionedStrength:        item.SanctionedStrength,
			Remarks:                   item.Remarks,
			ApproveStatus:             item.ApproveStatus,
			IsHeadOfTheOffice:         item.IsHeadOfTheOffice,
			EmployeeID:                item.EmployeeID,
			EmployeeName:              item.EmployeeName,
			// New authority fields
			EmployeePostID:          item.EmployeePostID,
			LeaveSancAuthority1:     item.LeaveSancAuthority1,
			LeaveSancAuthority2:     item.LeaveSancAuthority2,
			PayApproveAuthority1:    item.PayApproveAuthority1,
			AppointingAuthority:     item.AppointingAuthority,
			DisciplinaryAuthority:   item.DisciplinaryAuthority,
			DDOAuthority:            item.DDOAuthority,
			EmployeeOfficeID:        item.EmployeeOfficeID,
			VigilanceMakerAuthority: item.VigilanceMakerAuthority,
		}
		response = append(response, resp)
	}
	return response
}

type FetchPostsByOfficeIDAllAPIResponse2 struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []FetchPostsByOfficeIDResponseNew `json:"data"`
}

type PostDetails1Response struct {
	PostID       int32  `json:"post_id" db:"post_id"`
	PostName     string `json:"post_name" db:"post_name"`
	GroupId      int32  `json:"group_id" db:"group_id"`
	Designation  string `json:"designation" db:"designation"`
	PostStatus   string `json:"post_status" db:"post_status"`
	FilledStatus string `json:"filled_status" db:"status"`
}

func NewPostDetails1(data []domain.PostDetails1) []PostDetails1Response {
	var response []PostDetails1Response
	for _, item := range data {
		resp := PostDetails1Response{
			PostID:       item.PostID.Int32,
			PostName:     item.PostName.String,
			GroupId:      item.GroupId.Int32,
			Designation:  item.Designation.String,
			PostStatus:   item.PostStatus.String,
			FilledStatus: item.FilledStatus.String,
		}
		response = append(response, resp)
	}
	return response
}

type GetPostDetails1APIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []PostDetails1Response `json:"data"`
}

type ListAllCadresD1Response struct {
	CadreID   int       `json:"cadre_id"`
	CadreName string    `json:"cadre_name"`
	GroupName string    `json:"group_name"`
	PayLevel  int       `json:"pay_level"`
	GradePay  int       `json:"grade_pay"`
	ValidFrom time.Time `json:"valid_from"`
	ValidTo   time.Time `json:"valid_to"`
	Status    string    `json:"status"`
	Remarks   string    `json:"remarks"`
	GroupId   int       `json:"group_id"`
}

func NewListAllCadresD1Response(data []domain.CadreMasterD1) []ListAllCadresD1Response {
	var response []ListAllCadresD1Response
	for _, cadre := range data {
		allCadreResponse := ListAllCadresD1Response{
			CadreID:   cadre.CadreID,
			CadreName: cadre.CadreName,
			GroupName: cadre.GroupName,
			PayLevel:  cadre.PayLevel,
			GradePay:  cadre.GradePay,
			ValidFrom: cadre.ValidFrom,
			ValidTo:   cadre.ValidTo,
			Status:    cadre.Status,
			Remarks:   cadre.Remarks,
			GroupId:   cadre.GroupID,
		}
		response = append(response, allCadreResponse)
	}
	return response
}

type ListAllCadresD1APIResponse struct {
	port.StatusCodeAndMessage `json:",inline"`
	port.MetaDataResponse     `json:",inline"`
	Data                      []ListAllCadresD1Response `json:"data"`
}
