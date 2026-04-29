package handler

import (
	"pmdm/core/domain"
	"time"
)

// stringToUint64 is a helper function to convert a string to uint64
// func stringToUint64(str string) (uint64, error) {
// 	num, err := strconv.ParseUint(str, 10, 64)

// 	return num, err
// }

// // toMap is a helper function to add meta and data to a map
// func toMap(m meta, data any, key string) map[string]any {
// 	return map[string]any{
// 		"meta": m,
// 		key:    data,
// 	}
// }

// type SavePostRedeploymentRequest struct {
// 	PostID                    int       `json:"post_id" validate:"required"`
// 	CadreName                 string    `json:"cadre_name" validate:"required"`
// 	RedeploymentFromOfficeID  int       `json:"redeployment_from_office_id" validate:"required"`
// 	RedeploymentToOfficeID    int       `json:"redeployment_to_office_id" validate:"required"`
// 	RedeploymentToOfficeName  string    `json:"redeployment_to_office_name"`
// 	RedeploymentOn            time.Time `json:"redeployment_on" time_format:"2006-01-02"`
// 	RedeploymentBy            string    `json:"redeployment_by" validate:"required"`
// 	EffectiveFrom             time.Time `json:"effective_from" time_format:"2006-01-02"`
// 	EffectiveUPTo             time.Time `json:"effective_upto" time_format:"2006-01-02"`
// 	AllowancesAttached        bool      `json:"allowances_attached"`
// 	AllowanceDescription      string    `json:"allowance_description"`
// 	Remarks                   string    `json:"remarks"`
// 	OrderCasemark             string    `json:"order_casemark"`
// 	OrderDate                 time.Time `json:"order_date" time_format:"2006-01-02"`
// 	UploadOrderDocName        string    `json:"upload_order_doc_name"`
// 	EstablishmentRegisterID   int       `json:"establishment_register_id"`
// 	Designation               string    `json:"designation"`
// 	PermanentStatus           string    `json:"permanent_status"`
// 	EstablishmentRegisterName string    `json:"establishment_register_name"`
// 	GroupName                 string    `json:"group_name"`
// 	OfficeType                string    `json:"office_type"`
// 	OfficeSupervisor          string    `json:"office_supervisor"`
// 	IsHeadOfTheOffice         bool      `json:"is_head_of_the_office"`
// }

type SavePostRedeploymentRequest struct {
	PostID                    int    `json:"post_id"`
	CadreName                 string `json:"cadre_name"`
	RedeploymentFromOfficeID  int    `json:"redeployment_from_office_id"`
	RedeploymentToOfficeID    int    `json:"redeployment_to_office_id"`
	RedeploymentToOfficeName  string `json:"redeployment_to_office_name"`
	RedeploymentOn            string `json:"redeployment_on"` // or time.Time with custom unmarshal
	RedeploymentBy            string `json:"redeployment_by"`
	EffectiveFrom             string `json:"effective_from"` // use time.Time if working with dates
	EffectiveUPTo             string `json:"effective_to"`
	AllowancesAttached        bool   `json:"allowances_attached"`
	AllowanceDescription      string `json:"allowance_description"`
	Remarks                   string `json:"remarks"`
	OrderCasemark             string `json:"order_casemark"`
	OrderDate                 string `json:"order_date"`
	UploadOrderDocName        string `json:"upload_order_doc_name"`
	EstablishmentRegisterID   int    `json:"establishment_register_id"`
	Designation               string `json:"designation"`
	PermanentStatus           string `json:"permanent_status"`
	EstablishmentRegisterName string `json:"establishment_register_name"`
	GroupID                   int    `json:"group_id"`
	GroupName                 string `json:"group_name"`
	OfficeType                string `json:"office_type"`
	OfficeSupervisor          bool   `json:"office_supervisor"`
	IsHeadOfTheOffice         bool   `json:"is_head_of_the_office"`
	CadreID                   int    `json:"cadre_id"`
	DesignationID             int    `json:"designation_id"`
	PostName                  string `json:"post_name"`
}

func ToPostRedeploymentLog(req SavePostRedeploymentRequest) domain.PostRedeploymentLog {
	return domain.PostRedeploymentLog{
		PostID:                   req.PostID,
		CadreName:                req.CadreName,
		RedeploymentFromOfficeID: req.RedeploymentFromOfficeID,
		RedeploymentToOfficeID:   req.RedeploymentToOfficeID,
		RedeploymentOn:           req.RedeploymentOn,
		RedeploymentBy:           req.RedeploymentBy,
		EffectiveFrom:            req.EffectiveFrom,
		EffectiveUPTo:            req.EffectiveUPTo,
	}
}

func ToPostManagementMasterUpdate(req SavePostRedeploymentRequest) domain.PostManagementMasterUpdate {
	return domain.PostManagementMasterUpdate{
		OfficeID:                  req.RedeploymentToOfficeID,
		OfficeName:                req.RedeploymentToOfficeName,
		FilledStatus:              "Vacant",
		PostStatus:                "Active",
		AllowancesAttached:        req.AllowancesAttached,
		AllowanceDescription:      req.AllowanceDescription,
		UpdatedBy:                 req.RedeploymentBy,
		UpdatedDate:               req.RedeploymentOn,
		Status:                    "Active",
		Remarks:                   req.Remarks,
		ValidFrom:                 req.EffectiveFrom,
		ValidTo:                   req.EffectiveUPTo,
		OrderCasemark:             req.OrderCasemark,
		OrderDate:                 req.OrderDate,
		UploadOrderDocName:        req.UploadOrderDocName,
		EstablishmentRegisterID:   req.EstablishmentRegisterID,
		Designation:               req.Designation,
		PermanentStatus:           req.PermanentStatus,
		EstablishmentRegisterName: req.EstablishmentRegisterName,
		GroupName:                 req.GroupName,
		OfficeType:                req.OfficeType,
		OfficeSupervisor:          req.OfficeSupervisor,
		IsHeadOfTheOffice:         req.IsHeadOfTheOffice,
		CadreID:                   req.CadreID,
		DesignationID:             req.DesignationID,
		PostName:                  req.PostName,
	}
}

type UpdatePostDetailsbyPostIDRequest struct {
	ApprovePostID      string    `json:"approve_post_id" validate:"required"`
	Cadre              int       `json:"cadre_id" validate:"required"`
	CadreName          string    `json:"cadre_name" validate:"required"`
	Designation        string    `json:"designation" validate:"required"`
	DesignationID      int       `json:"designation_id" validate:"required"`
	CareatedBy         string    `json:"created_by"`
	EmployeeGroup      string    `json:"employee_group" validate:"required"`
	GradePay           int       `json:"grade_pay"`
	GroupID            int       `json:"group_id" validate:"required"`
	OfficeID           int       `json:"office_id" validate:"required"`
	Office             string    `json:"office_name" validate:"required"`
	OrderCaseMark      string    `json:"order_casemark" validate:"required"`
	OrderDate          time.Time `json:"order_date" validate:"required"`
	PayLevel           int       `json:"pay_level"`
	PostID             int       `json:"post_id" validate:"required"`
	PostName           string    `json:"post_name" validate:"required"`
	Remarks            string    `json:"remarks" validate:"required"`
	Status             string    `json:"status" validate:"required"`
	UploadOrderDocName string    `json:"upload_order_doc_name"`
}

func ToPostManagementMasterUpdate2(req UpdatePostDetailsbyPostIDRequest) domain.UpdatePostManagementMaster {
	return domain.UpdatePostManagementMaster{
		ApprovePostID:      req.ApprovePostID,
		CadreID:            req.Cadre,
		CadreName:          req.CadreName,
		Designation:        req.Designation,
		DesignationID:      req.DesignationID,
		CreatedBy:          req.CareatedBy,
		EmployeeGroup:      req.EmployeeGroup,
		GradePay:           req.GradePay,
		GroupID:            req.GroupID,
		OfficeID:           req.OfficeID,
		OfficeName:         req.Office,
		OrderCaseMark:      req.OrderCaseMark,
		OrderDate:          req.OrderDate,
		PayLevel:           req.PayLevel,
		PostID:             req.PostID,
		PostName:           req.PostName,
		Remarks:            req.Remarks,
		Status:             req.Status,
		UploadOrderDocName: req.UploadOrderDocName,
	}
}
