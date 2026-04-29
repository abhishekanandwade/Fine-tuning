package domain

import (
	"time"

	"github.com/volatiletech/null/v9"
)

// PostManagementMaster represents a record from the postmanagement_master table
// type PostManagementMaster struct {
// 	PostManagementID          int       `json:"postmanagement_id"`
// 	OfficeID                  int       `json:"office_id"`
// 	PostID                    int       `json:"post_id"`
// 	PostName                  string    `json:"post_name"`
// 	OfficeName                string    `json:"office_name"`
// 	GroupId                   int       `json:"group_id"`
// 	GroupName                 string    `json:"group_name"`
// 	CadreName                 string    `json:"cadre_name"`
// 	FilledStatus              string    `json:"filled_status"`
// 	PostStatus                string    `json:"post_status"`
// 	AllowancesAttached        bool      `json:"allowances_attached"`
// 	AllowanceDescription      string    `json:"allowance_description"`
// 	CreatedBy                 string    `json:"created_by"`
// 	CreatedOn                 time.Time `json:"created_on"`
// 	ApprovedBy                string    `json:"approved_by"`
// 	ApprovedOn                time.Time `json:"approved_on"`
// 	UpdatedBy                 string    `json:"updated_by"`
// 	UpdatedOn                 time.Time `json:"updated_on"`
// 	Status                    string    `json:"status"`
// 	Remarks                   string    `json:"remarks"`
// 	ValidFrom                 time.Time `json:"valid_from"`
// 	ValidTo                   time.Time `json:"valid_to"`
// 	OrderCaseMark             string    `json:"order_casemark"`
// 	OrderDate                 time.Time `json:"order_date"`
// 	UploadOrderDocName        string    `json:"upload_order_doc_name"`
// 	EstablishmentRegisterID   int       `json:"establishment_register_id"`
// 	Designation               string    `json:"designation"`
// 	PayLevel                  int       `json:"pay_level"`
// 	GradePay                  int       `json:"grade_pay"`
// 	PermanentStatus           bool      `json:"permanent_status"`
// 	EstablishmentRegisterName string    `json:"establishment_register_name"`
// 	EmployeeGroup             string    `json:"employee_group" `
// 	SanctionedStrength        int       `json:"sanctioned_strength"`
// 	Count                     int       `json:"count"`
// 	CadreID                   int       `json:"cadre_id"`
// 	DesignationId             int       `json:"designation_id"`
// 	ApprovePostID             string    `json:"approve_post_id"`
// }

type PostManagementMaster struct {
	PostManagementID          null.Int32  `json:"postmanagement_id" db:"postmanagement_id"`
	OfficeID                  null.Int32  `json:"office_id" db:"office_id"`
	PostID                    null.Int32  `json:"post_id" db:"post_id"`
	PostName                  null.String `json:"post_name" db:"post_name"`
	OfficeName                null.String `json:"office_name" db:"office_name"`
	GroupId                   null.Int32  `json:"group_id" db:"group_id"`
	GroupName                 null.String `json:"group_name" db:"group_name"`
	CadreName                 null.String `json:"cadre_name" db:"cadre_name"`
	FilledStatus              null.String `json:"filled_status" db:"filled_status"`
	PostStatus                null.String `json:"post_status" db:"post_status"`
	AllowancesAttached        null.Bool   `json:"allowances_attached" db:"allowances_attached"`
	AllowanceDescription      null.String `json:"allowance_description" db:"allowance_description"`
	CreatedBy                 null.String `json:"created_by" db:"created_by"`
	CreatedOn                 null.Time   `json:"created_on" db:"created_date"`
	ApprovedBy                null.String `json:"approved_by" db:"approved_by"`
	ApprovedOn                null.Time   `json:"approved_on" db:"approved_date"`
	UpdatedBy                 null.String `json:"updated_by" db:"updated_by"`
	UpdatedOn                 null.Time   `json:"updated_on" db:"updated_date"`
	Status                    null.String `json:"status" db:"status"`
	Remarks                   null.String `json:"remarks" db:"remarks"`
	ValidFrom                 null.Time   `json:"valid_from" db:"valid_from"`
	ValidTo                   null.Time   `json:"valid_to" db:"valid_to"`
	OrderCaseMark             null.String `json:"order_casemark" db:"order_casemark"`
	OrderDate                 null.Time   `json:"order_date" db:"order_date"`
	UploadOrderDocName        null.String `json:"upload_order_doc_name" db:"upload_order_doc_name"`
	EstablishmentRegisterID   null.Int32  `json:"establishment_register_id" db:"establishment_register_id"`
	Designation               null.String `json:"designation" db:"designation"`
	PayLevel                  null.Int32  `json:"pay_level" db:"pay_level"`
	GradePay                  null.Int32  `json:"grade_pay" db:"grade_pay"`
	PermanentStatus           null.Bool   `json:"permanent_status" db:"permanent_status"`
	EstablishmentRegisterName null.String `json:"establishment_register_name" db:"establishment_register_name"`
	EmployeeGroup             null.String `json:"employee_group" db:"employee_group"`
	SanctionedStrength        null.Int32  `json:"sanctioned_strength" db:"sanctioned_strength"`
	Count                     null.Int32  `json:"count" db:"count"`
	CadreID                   null.Int32  `json:"cadre_id" db:"cadre_id"`
	DesignationId             null.Int32  `json:"designation_id" db:"designation_id"`
	ApprovePostID             null.String `json:"approve_post_id" db:"approve_post_id"`
	ApproveStatus             null.String `json:"approve_status" db:"approve_status"`
	EmployeeID                null.Int32  `json:"employee_id" db:"employee_id"`     // Employee ID from kafka_employee_master
	EmployeeName              null.String `json:"employee_name" db:"employee_name"` // Employee full name (concatenated)
}
type PostManagementMaster1 struct {
	EstablishmentRegisterID   int       `json:"establishment_register_id"`
	OfficeID                  int       `json:"office_id"`
	OfficeName                string    `json:"office_name"`
	EstablishmentRegisterName string    `json:"establishment_register_name"`
	CreatedBy                 string    `json:"created_by"`
	CreatedOn                 time.Time `json:"created_date"`
	Status                    string    `json:"status"`
}

type PostManagementMaster2 struct {
	PostManagementID int64  `db:"postmanagement_id" json:"postmanagement_id"`
	OfficeID         int64  `db:"office_id" json:"office_id"`
	PostID           int64  `db:"post_id" json:"post_id"`
	PostName         string `db:"post_name" json:"post_name"`
	OfficeName       string `db:"office_name" json:"office_name"`
	Status           string `db:"status" json:"status"`
}

type PostManagementMaster3 struct {
	PostManagementMakerID int    `db:"postmanagement_maker_id" json:"postmanagement_maker_id"`
	OfficeID              int64  `db:"office_id" json:"office_id"`
	PostID                int64  `db:"post_id" json:"post_id"`
	PostName              string `db:"post_name" json:"post_name"`
	OfficeName            string `db:"office_name" json:"office_name"`
	Status                string `db:"status" json:"status"`
}

type PostManagementMaster4 struct {
	OfficeID int64  `db:"office_id" json:"office_id"`
	PostID   int64  `db:"post_id" json:"post_id"`
	Status   string `db:"status" json:"status"`
}
type DocumentMaster struct {
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
	DocumentUpdatedBy      string    `json:"document_updated_by"`
	DocumentUpdatedDate    time.Time `json:"document_updated_date"`
	DocumentApprovedBy     string    `json:"document_approved_by"`
	DocumentApprovedDate   time.Time `json:"document_approved_date"`
	Remarks                string    `json:"remarks"`
	DocumentFilePath       string    `json:"document_file_path"`
}

type Document struct {
	//PostID               int       `db:"post_id"`
	OfficeID             int       `db:"office_id"`
	DocumentName         string    `db:"document_name"`
	DocumentType         string    `db:"document_type"`
	DocumentSize         int64     `db:"document_size"`
	DocumentFilePath     string    `db:"document_file_path"`
	DocumentUploadStatus string    `db:"document_upload_status"`
	DocumentUploadedBy   string    `db:"document_uploaded_by"`
	DocumentUploadedDate time.Time `db:"document_uploaded_date"`
	//ApproverPostID       string    `db:"document_approver_post_id"`
}

//	type PostManagementMaker struct {
//		PostManagementMakerID     int       `json:"postmanagement_maker_id"`
//		OfficeID                  int       `json:"office_id"`
//		PostID                    int       `json:"post_id"`
//		PostName                  string    `json:"post_name"`
//		OfficeName                string    `json:"office_name"`
//		NewOfficeID               int       `json:"new_office_id" validate:"required"`
//		NewOfficeName             string    `json:"new_office_name" validate:"required"`
//		GroupId                   int       `json:"group_id"`
//		CadreName                 string    `json:"cadre_name"`
//		FilledStatus              string    `json:"filled_status"`
//		PostStatus                string    `json:"post_status"`
//		AllowancesAttached        bool      `json:"allowances_attached"`
//		AllowanceDescription      string    `json:"allowance_description"`
//		CreatedBy                 string    `json:"created_by"`
//		CreatedDate               time.Time `json:"created_date"`
//		Status                    string    `json:"status"`
//		ValidFrom                 time.Time `json:"valid_from"`
//		ValidTo                   time.Time `json:"valid_to"`
//		OrderCaseMark             string    `json:"order_case_mark"`
//		OrderDate                 time.Time `json:"order_date"`
//		UploadOrderDocName        string    `json:"upload_order_doc_name"`
//		EstablishmentRegisterID   int       `json:"establishment_register_id"`
//		Designation               string    `json:"designation"`
//		PayLevel                  int       `json:"pay_level"`
//		GradePay                  int       `json:"grade_pay"`
//		PermanentStatus           bool      `json:"permanent_status"`
//		EstablishmentRegisterName string    `json:"establishment_register_name"`
//		EmployeeGroup             string    `json:"employee_group"`
//		SanctionedStrength        int       `json:"sanctioned_strength"`
//		Remarks                   string    `json:"remarks"`
//		DesignationId             int       `json:"designation_id"`
//		CadreID                   int       `json:"cadre_id"`
//		ApproveStatus             string    `json:"approve_status"`
//		ApprovePostID             string    `json:"approve_post_id"`
//		GroupName                 string    `json:"group_name"`
//	}
type PostManagementMaker struct {
	PostManagementMakerID     int       `json:"postmanagement_maker_id" db:"postmanagement_maker_id"`
	OfficeID                  int       `json:"office_id" db:"office_id"`
	PostID                    int       `json:"post_id" db:"post_id"`
	PostName                  string    `json:"post_name" db:"post_name"`
	OfficeName                string    `json:"office_name" db:"office_name"`
	NewOfficeID               int       `json:"new_office_id" db:"new_office_id" `
	NewOfficeName             string    `json:"new_office_name" db:"new_office_name" `
	GroupId                   int       `json:"group_id" db:"group_id"`
	CadreName                 string    `json:"cadre_name" db:"cadre_name"`
	FilledStatus              string    `json:"filled_status" db:"filled_status"`
	PostStatus                string    `json:"post_status" db:"post_status"`
	AllowancesAttached        bool      `json:"allowances_attached" db:"allowances_attached"`
	AllowanceDescription      string    `json:"allowance_description" db:"allowance_description"`
	CreatedBy                 string    `json:"created_by" db:"created_by"`
	CreatedDate               time.Time `json:"created_date" db:"created_date"`
	Status                    string    `json:"status" db:"status"`
	ValidFrom                 time.Time `json:"valid_from" db:"valid_from"`
	ValidTo                   time.Time `json:"valid_to" db:"valid_to"`
	OrderCaseMark             string    `json:"order_case_mark" db:"order_casemark"`
	OrderDate                 time.Time `json:"order_date" db:"order_date"`
	UploadOrderDocName        string    `json:"upload_order_doc_name" db:"upload_order_doc_name"`
	EstablishmentRegisterID   int       `json:"establishment_register_id" db:"establishment_register_id"`
	Designation               string    `json:"designation" db:"designation"`
	PayLevel                  int       `json:"pay_level" db:"pay_level"`
	GradePay                  int       `json:"grade_pay" db:"grade_pay"`
	PermanentStatus           bool      `json:"permanent_status" db:"permanent_status"`
	EstablishmentRegisterName string    `json:"establishment_register_name" db:"establishment_register_name"`
	EmployeeGroup             string    `json:"employee_group" db:"employee_group"`
	SanctionedStrength        int       `json:"sanctioned_strength" db:"sanctioned_strength"`
	Remarks                   string    `json:"remarks" db:"remarks"`
	DesignationId             int       `json:"designation_id" db:"designation_id"`
	CadreID                   int       `json:"cadre_id" db:"cadre_id"`
	ApproveStatus             string    `json:"approve_status" db:"approve_status"`
	ApprovePostID             string    `json:"approve_post_id" db:"approve_post_id"`
	GroupName                 string    `json:"group_name"`
	ExchangePostID            int       `json:"exchange_post_id" db:"exchange_post_id"`
	//MasterMakerID             string    `json:"master_maker_id" db:"master_maker_id"`
}
type PostManagementMakerApproveResponse struct {
	ApprovedPosts []int `json:"approved_posts"`
}

type MasterMakerID struct {
	MasterMakerID string `json:"master_maker_id" `
}

type PostManagementMaster5 struct {
	PostName string `db:"post_name" json:"post_name"`
	Group    string `db:"group" json:"group"`
	Cadre    string `db:"cadre" json:"cadre"`
}

type PostManagementMaker1 struct {
	OfficeID        int       `db:"office_id" json:"office_id"`
	PostName        string    `db:"post_name" json:"post_name"`
	OfficeName      string    `db:"office_name" json:"office_name"`
	GroupID         int       `db:"group_id" json:"group_id"`
	FilledStatus    string    `db:"filled_status" json:"filled_status"`
	PostID          int       `db:"post_id" json:"post_id"`
	Designation     string    `db:"designation" json:"designation"`
	PermanentStatus bool      `db:"permanent_status" json:"permanent_status"`
	ApproveStatus   string    `db:"approve_status" json:"approve_status"`
	Status          string    `db:"status" json:"status"`
	NewOfficeID     int       `db:"new_office_id" json:"new_office_id"`
	NewOfficeName   string    `db:"new_office_name" json:"new_office_name"`
	Remarks         string    `db:"remarks" json:"remarks"`
	ExchangePostID  int       `db:"exchange_post_id" json:"exchange_post_id"`
	OrderDate       time.Time `db:"order_date" json:"order_date"`
	EmployeeGroup   string    `json:"employee_group" db:"employee_group"`
	CadreID         int       `json:"cadre_id" db:"cadre_id"`
	CadreName       string    `json:"cadre_name" db:"cadre_name"`
	PayLevel        int       `json:"pay_level" db:"pay_level"`

	// Old record fields from the main table
	OldCadreID       int    `db:"old_cadre_id" json:"old_cadre_id"`
	OldCadreName     string `db:"old_cadre_name" json:"old_cadre_name"`
	OldGroupID       int    `db:"old_group_id" json:"old_group_id"`
	OldDesignation   string `db:"old_designation" json:"old_designation"`
	OldPayLevel      string `db:"old_pay_level" json:"old_pay_level"`
	OldGradePay      string `db:"old_grade_pay" json:"old_grade_pay"`
	OldStatus        string `db:"old_status" json:"old_status"`                 // Added old status
	OldEmployeeGroup string `db:"old_employee_group" json:"old_employee_group"` // Old employee group
	OldPostName      string `db:"old_post_name" json:"old_post_name"`
}

type ListManagementMaker struct {
	CircleName              null.String `json:"circle_name" db:"circle_name"`
	CircleOfficeID          null.Int    `json:"circle_office_id" db:"circle_office_id"`
	RegionName              null.String `json:"region_name" db:"region_name"`
	RegionOfficeID          null.Int    `json:"region_office_id" db:"region_office_id"`
	DivisionName            null.String `json:"division_name" db:"division_name"`
	DivisionOfficeID        null.Int    `json:"division_office_id" db:"division_office_id"`
	SubDivisionName         null.String `json:"sub_division_name" db:"sub_division_name"`
	SubDivisionOfficeID     null.Int    `json:"sub_division_office_id" db:"sub_division_office_id"`
	HoID                    null.Int    `json:"ho_id" db:"ho_id"`
	HoName                  null.String `json:"ho_name" db:"ho_name"`
	HroID                   null.Int    `json:"hro_id" db:"hro_id"`
	HroName                 null.String `json:"hro_name" db:"hro_name"`
	SoID                    null.Int    `json:"so_id" db:"so_id"`
	SoName                  null.String `json:"so_name" db:"so_name"`
	SroID                   null.Int    `json:"sro_id" db:"sro_id"`
	SroName                 null.String `json:"sro_name" db:"sro_name"`
	BoID                    null.Int    `json:"bo_id" db:"bo_id"`
	BoName                  null.String `json:"bo_name" db:"bo_name"`
	OfficeName              null.String `json:"office_name" db:"office_name"`
	OfficeID                null.Int    `json:"office_id" db:"office_id"`
	PostID                  null.Int    `json:"post_id" db:"post_id"`
	PostName                null.String `json:"post_name" db:"post_name"`
	GroupID                 null.Int    `json:"group_id" db:"group_id"`
	EmployeeGroup           null.String `json:"employee_group" db:"employee_group"`
	CadreID                 null.Int    `json:"cadre_id" db:"cadre_id"`
	CadreName               null.String `json:"cadre_name" db:"cadre_name"`
	FilledStatus            null.String `json:"filled_status" db:"filled_status"`
	EstablishmentRegisterID null.Int    `json:"establishment_register_id" db:"establishment_register_id"`
	Designation             null.String `json:"designation" db:"designation"`
	PayLevel                null.Int    `json:"pay_level" db:"pay_level"`
	GradePay                null.Int    `json:"grade_pay" db:"grade_pay"`
	UpdateDate              null.String `json:"updated_date" db:"updated_date"`
}

type ListAvailablePosts struct {
	OfficeID      null.Int    `json:"office_id" db:"office_id"`
	OfficeName    null.String `json:"office_name" db:"office_name"`
	PostID        null.Int    `json:"post_id" db:"post_id"`
	PostName      null.String `json:"post_name" db:"post_name"`
	GroupID       null.Int    `json:"group_id" db:"group_id"`
	CadreName     null.String `json:"cadre_name" db:"cadre_name"`
	Designation   null.String `json:"designation" db:"designation"`
	EmployeeGroup null.String `json:"employee_group" db:"employee_group"`
	PostStatus    null.String `json:"post_status" db:"post_status"`
	FilledStatus  null.String `json:"filled_status" db:"filled_status"`
}

type ListGroupMaster struct {
	GroupID   int    `json:"group_id"`
	GroupName string `json:"group_name"`
}

type ListOfficeDetails struct {
	CadreName                *string `json:"cadre_name"`
	GroupName                *string `json:"group_name"`
	PostID                   *string `json:"post_id"`
	Designation              *string `json:"designation"`
	OfficeID                 string  `json:"office_id"`
	DivisionOfficeID         *string `json:"division_office_id"`
	CircleOfficeID           *string `json:"circle_office_id"`
	RegionOfficeID           *string `json:"region_office_id"`
	ReportingOfficeID        *string `json:"reporting_office_id"`
	ReportingAuthorityPostID *string `json:"reporting_authority_post_id"`
	CadreID                  *string `json:"cadre_id"`
	DesignationID            *string `json:"designation_id"`
}

type ListGroupCadre struct {
	CadreID   *string `json:"cadre_id"`
	CadreName *string `json:"cadre_name"`
	PayLevel  *string `json:"pay_level"`
	GradePay  *string `json:"grade_pay"`
	GroupID   string  `json:"group_id"`
	GroupName *string `json:"group_name"`
}
type CreatePostRequest struct {
	OfficeID      int64   `json:"office_id" validate:"required"`
	PostName      string  `json:"post_name" validate:"required"`
	OfficeName    string  `json:"office_name" validate:"required"`
	GroupID       int64   `json:"group_id" validate:"required"`
	CadreName     string  `json:"cadre_name" validate:"required"`
	CreatedBy     string  `json:"created_by" validate:"required"`
	Designation   string  `json:"designation" validate:"required"`
	PayLevel      int64   `json:"pay_level" validate:"required"`
	GradePay      string  `json:"grade_pay" validate:"required"`
	EmployeeGroup string  `json:"employee_group" validate:"required"`
	GroupName     string  `json:"group_name" validate:"required"`
	CadreID       int64   `json:"cadre_id" validate:"required"`
	DesignationID int64   `json:"designation_id" validate:"required"`
	ApprovePostID *int64  `json:"approve_post_id"`
	AdminOfficeID *int64  `json:"admin_office_id"`
	EmployeeType  *string `json:"employee_type"`
	OfficeType    *string `json:"office_type"`
	LoginID       *string `json:"login_id"`
	NumberOfPosts int     `json:"number_of_posts" validate:"required,min=1"`
}

type UpdatePostRequest struct {
	OfficeID      int64  `json:"office_id" validate:"required"`
	OfficeName    string `json:"office_name" validate:"required"`
	AdminOfficeID int64  `json:"admin_office_id" validate:"required"`
	OfficeType    string `json:"office_type" validate:"required"`
	PostId        int64  `json:"post_id" validate:"required"`
}

type VacantPost struct {
	PostID int64 `json:"post_id"`
}

type PostDetails struct {
	OfficeID          int64       `json:"office_id"`
	OfficeName        null.String `json:"office_name"`
	PostID            int64       `json:"post_id"`
	PostName          null.String `json:"post_name"`
	GroupID           null.Int64  `json:"group_id"`
	GroupName         null.String `json:"group_name"`
	CadreID           null.Int64  `json:"cadre_id"`
	CadreName         null.String `json:"cadre_name"`
	Designation       null.String `json:"designation"`
	DesignationID     null.Int64  `json:"designation_id"`
	IsHeadOfTheOffice null.Bool   `json:"is_head_of_the_office"`
	EmployeeID        null.Int64  `json:"employee_id"`
	EmployeeName      null.String `json:"employee_name"`
}
type PostManagementMasterNew struct {
	PostManagementID          int       `json:"postmanagement_id" db:"postmanagement_id"`
	OfficeID                  int       `json:"office_id" db:"office_id"`
	PostID                    int       `json:"post_id" db:"post_id"`
	PostName                  string    `json:"post_name" db:"post_name"`
	OfficeName                string    `json:"office_name" db:"office_name"`
	GroupId                   int       `json:"group_id" db:"group_id"`
	GroupName                 string    `json:"group_name" db:"group_name"`
	CadreName                 string    `json:"cadre_name" db:"cadre_name"`
	FilledStatus              string    `json:"filled_status" db:"filled_status"`
	PostStatus                string    `json:"post_status" db:"post_status"`
	AllowancesAttached        bool      `json:"allowances_attached" db:"allowances_attached"`
	AllowanceDescription      string    `json:"allowance_description" db:"allowance_description"`
	CreatedBy                 string    `json:"created_by" db:"created_by"`
	CreatedDate               time.Time `json:"created_on" db:"created_date"`
	ApprovedBy                string    `json:"approved_by" db:"approved_by"`
	ApprovedDate              time.Time `json:"approved_on" db:"approved_date"`
	UpdatedBy                 string    `json:"updated_by" db:"updated_by"`
	UpdatedDate               time.Time `json:"updated_on" db:"updated_date"`
	Status                    string    `json:"status" db:"status"`
	Remarks                   string    `json:"remarks" db:"remarks"`
	ValidFrom                 time.Time `json:"valid_from" db:"valid_from"`
	ValidTo                   time.Time `json:"valid_to" db:"valid_to"`
	OrderCaseMark             string    `json:"order_casemark" db:"order_casemark"`
	OrderDate                 time.Time `json:"order_date" db:"order_date"`
	UploadOrderDocName        string    `json:"upload_order_doc_name" db:"upload_order_doc_name"`
	EstablishmentRegisterID   int       `json:"establishment_register_id" db:"establishment_register_id"`
	Designation               string    `json:"designation" db:"designation"`
	PayLevel                  int32     `json:"pay_level" db:"pay_level"`
	GradePay                  int32     `json:"grade_pay" db:"grade_pay"`
	PermanentStatus           bool      `json:"permanent_status" db:"permanent_status"`
	EstablishmentRegisterName string    `json:"establishment_register_name" db:"establishment_register_name"`
	EmployeeGroup             string    `json:"employee_group" db:"employee_group"`
	SanctionedStrength        int       `json:"sanctioned_strength" db:"sanctioned_strength"`
	Count                     int32     `json:"count" db:"count"`
	CadreID                   int       `json:"cadre_id" db:"cadre_id"`
	DesignationId             int       `json:"designation_id" db:"designation_id"`
	ApprovePostID             string    `json:"approve_post_id" db:"approve_post_id"`
	ApproveStatus             string    `json:"approve_status" db:"approve_status"`
	IsHeadOfTheOffice         bool      `json:"is_head_of_the_office" db:"is_head_of_the_office"`

	// From kafka_employee_master
	EmployeeID   int32  `json:"employee_id" db:"employee_id"`
	EmployeeName string `json:"employee_name" db:"employee_name"`

	// From post_mapping_detail (selected fields only)
	EmployeePostID          int32  `json:"employee_post_id" db:"employee_post_id"`
	LeaveSancAuthority1     string `json:"leave_sanc_authority_1" db:"leave_sanc_authority_1"`
	LeaveSancAuthority2     string `json:"leave_sanc_authority_2" db:"leave_sanc_authority_2"`
	PayApproveAuthority1    string `json:"pay_approve_authority1" db:"pay_approve_authority1"`
	AppointingAuthority     string `json:"appointing_authority" db:"appointing_authority"`
	DisciplinaryAuthority   string `json:"disciplinary_authority" db:"disciplinary_authority"`
	DDOAuthority            string `json:"ddo_authority" db:"ddo_authority"`
	EmployeeOfficeID        int32  `json:"employee_office_id" db:"employee_office_id"`
	VigilanceMakerAuthority string `json:"vigilence_maker_authority" db:"vigilence_maker_authority"`
}
type HeadOfOfficeRequest struct {
	OfficeID int64 `json:"office_id" binding:"required"`
}

type HeadOfOfficeEntry struct {
	PostManagementID int64  `json:"postmanagement_id"`
	OfficeID         int64  `json:"office_id"`
	PostID           int64  `json:"post_id"`
	PostName         string `json:"post_name"`
	CadreName        string `json:"cadre_name"`
	RankByCadre      *int   `json:"rank_by_cadre,omitempty"`
	RankByPay        *int   `json:"rank_by_pay,omitempty"`
}

type HeadOfOfficeResponse struct {
	StatusCode int                 `json:"status_code"`
	Message    string              `json:"message"`
	Data       []HeadOfOfficeEntry `json:"data"`
}

// domain/head_of_office.go (new / updated)

type UpdateHeadOfOfficeRequest struct {
	OfficeID        int64 `json:"office_id"        binding:"required"`
	PostID          int64 `json:"post_id"          binding:"required"`  // new HoO
	RequestorPostID int64 `json:"requestor_post_id" binding:"required"` // caller’s post
}

type GenericSuccessResponse struct {
	StatusCode int    `json:"status_code"`
	Message    string `json:"message"`
}
type HeadPostOccupancy struct {
	EmployeeID   int64  `json:"employee_id"`
	EmployeeName string `json:"employee_name"`
	PostID       int64  `json:"post_id"`
	OfficeID     int64  `json:"office_id"`
	PostName     string `json:"post_name"`
	OfficeName   string `json:"office_name"`
	GroupName    string `json:"group_name"`
	CadreName    string `json:"cadre_name"`
	Status       string `json:"status"`
}

type PostRedeployment struct {
	OfficeID     null.Int    `json:"office_id" db:"office_id"`
	OfficeName   null.String `json:"office_name" db:"office_name"`
	OfficeType   null.String `json:"office_type" db:"office_type"`
	PostID       null.Int    `json:"post_id" db:"post_id"`
	PostName     null.String `json:"post_name" db:"post_name"`
	CadreName    null.String `json:"cadre_name" db:"cadre_name"`
	Designation  null.String `json:"designation" db:"designation"`
	EmployeeName null.String `json:"employee_name" db:"employee_name"`
	FilledStatus null.String `json:"filled_status" db:"filled_status"`
}

type PostRedeploymentLog struct {
	PostID                   int    `json:"post_id" db:"post_id"`
	CadreName                string `json:"cadre_name" db:"cadre_name"`
	RedeploymentFromOfficeID int    `json:"redeployment_from_office_id" db:"redeployment_from_office_id"`
	RedeploymentToOfficeID   int    `json:"redeployment_to_office_id" db:"redeployment_to_office_id"`
	RedeploymentOn           string `json:"redeployment_on" db:"redeployment_on"`
	RedeploymentBy           string `json:"redeployment_by" db:"redeployment_by"`
	EffectiveFrom            string `json:"effective_from" db:"effective_from"`
	EffectiveUPTo            string `json:"effective_to" db:"effective_upto"`
}

type PostManagementMasterUpdate struct {
	OfficeID                  int    `json:"office_id" db:"office_id"`
	OfficeName                string `json:"office_name" db:"office_name"`
	FilledStatus              string `json:"filled_status" db:"filled_status"`
	PostStatus                string `json:"post_status" db:"post_status"`
	AllowancesAttached        bool   `json:"allowances_attached" db:"allowances_attached"`
	AllowanceDescription      string `json:"allowance_description" db:"allowance_description"`
	UpdatedBy                 string `json:"updated_by" db:"updated_by"`
	UpdatedDate               string `json:"updated_date" db:"updated_date"`
	Status                    string `json:"status" db:"status"`
	Remarks                   string `json:"remarks" db:"remarks"`
	ValidFrom                 string `json:"valid_from" db:"valid_from"`
	ValidTo                   string `json:"valid_to" db:"valid_to"`
	OrderCasemark             string `json:"order_casemark" db:"order_casemark"`
	OrderDate                 string `json:"order_date" db:"order_date"`
	UploadOrderDocName        string `json:"upload_order_doc_name" db:"upload_order_doc_name"`
	EstablishmentRegisterID   int    `json:"establishment_register_id" db:"establishment_register_id"`
	Designation               string `json:"designation" db:"designation"`
	PayLevel                  string `json:"pay_level" db:"pay_level"`
	GradePay                  string `json:"grade_pay" db:"grade_pay"`
	PermanentStatus           string `json:"permanent_status" db:"permanent_status"`
	EstablishmentRegisterName string `json:"establishment_register_name" db:"establishment_register_name"`
	SanctionedStrength        int    `json:"sanctioned_strength" db:"sanctioned_strength"`
	GroupName                 string `json:"group_name" db:"group_name"`
	OfficeType                string `json:"office_type" db:"office_type"`
	OfficeSupervisor          bool   `json:"office_supervisor" db:"office_supervisor"`
	IsHeadOfTheOffice         bool   `json:"is_head_of_the_office" db:"is_head_of_the_office"`
	CadreID                   int    `json:"cadre_id" db:"cadre_id"`
	DesignationID             int    `json:"designation_id" db:"designation_id"`
	PostName                  string `json:"post_name" db:"post_name"`
}

type CircleName struct {
	CircleOfficeID int    `json:"circle_office_id" db:"circle_office_id"`
	CircleName     string `json:"circle_name" db:"circle_name"`
}

type RegionName struct {
	RegionOfficeID int    `json:"region_office_id" db:"region_office_id"`
	RegionName     string `json:"region_name" db:"region_name"`
}

type DivisionName struct {
	DivisionOfficeID int    `json:"division_office_id" db:"division_office_id"`
	DivisionName     string `json:"division_name" db:"division_name"`
}

type CadreName struct {
	CadreID   int    `json:"cadre_id" db:"cadre_id"`
	CadreName string `json:"cadre_name" db:"cadre_name"`
}

type PostIDDetails struct {
	OfficeID      null.Int    `json:"office_id" db:"office_id"`
	OfficeName    null.String `json:"office_name" db:"office_name"`
	PostID        null.Int    `json:"post_id" db:"post_id"`
	PostName      null.String `json:"post_name" db:"post_name"`
	GroupID       null.Int    `json:"group_id" db:"group_id"`
	GroupName     null.String `json:"group_name" db:"group_name"`
	CadreID       null.Int    `json:"cadre_id" db:"cadre_id"`
	CadreName     null.String `json:"cadre_name" db:"cadre_name"`
	DesignationID null.Int    `json:"designation_id" db:"designation_id"`
	Designation   null.String `json:"designation_name" db:"designation"`
	PostStatus    null.String `json:"post_status" db:"post_status"`
	EmployeeID    null.Int    `json:"employee_id" db:"employee_id"`
	EmployeeName  null.String `json:"employee_name" db:"employee_name"`
}

type PostsStatus struct {
	CircleOfficeID   null.Int    `json:"circle_office_id" db:"circle_office_id"`
	CircleName       null.String `json:"circle_name" db:"circle_name"`
	RegionOfficeID   null.Int    `json:"region_office_id" db:"region_office_id"`
	RegionName       null.String `json:"region_name" db:"region_name"`
	DivisionOfficeID null.Int    `json:"division_office_id" db:"division_office_id"`
	DivisionName     null.String `json:"division_name" db:"division_name"`
	OfficeID         null.Int    `json:"office_id" db:"office_id"`
	OfficeName       null.String `json:"office_name" db:"office_name"`
	GroupID          null.Int    `json:"group_id" db:"group_id"`
	GroupName        null.String `json:"group_name" db:"group_name"`
	CadreID          null.Int    `json:"cadre_id" db:"cadre_id"`
	CadreName        null.String `json:"cadre_name" db:"cadre_name"`
	TotalPosts       int         `json:"total_posts" db:"total_posts"`
	TotalFilledPosts int         `json:"total_filled_posts" db:"total_filled_posts"`
	TotalVacantPosts int         `json:"total_vacant_posts" db:"total_vacant_posts"`
}

type PostsCreatedRedeployedAbolished struct {
	CircleOfficeID   null.Int    `json:"circle_office_id" db:"circle_office_id"`
	CircleName       null.String `json:"circle_name" db:"circle_name"`
	RegionOfficeID   null.Int    `json:"region_office_id" db:"region_office_id"`
	RegionName       null.String `json:"region_name" db:"region_name"`
	DivisionOfficeID null.Int    `json:"division_office_id" db:"division_office_id"`
	DivisionName     null.String `json:"division_name" db:"division_name"`
	OfficeID         null.Int    `json:"office_id" db:"office_id"`
	PostsCreated     int         `json:"posts_created" db:"posts_created"`
	PostsRedeployed  int         `json:"posts_redeployed" db:"posts_redeployed"`
	PostsAbolished   int         `json:"posts_abolished" db:"posts_abolished"`
}

type PostsFilledVacantStatusDetailed struct {
	OfficeID      null.Int    `json:"office_id" db:"office_id"`
	OfficeName    null.String `json:"office_name" db:"office_name"`
	PostID        null.Int    `json:"post_id" db:"post_id"`
	PostName      null.String `json:"post_name" db:"post_name"`
	GroupID       null.Int    `json:"group_id" db:"group_id"`
	GroupName     null.String `json:"group_name" db:"group_name"`
	CadreID       null.Int    `json:"cadre_id" db:"cadre_id"`
	CadreName     null.String `json:"cadre_name" db:"cadre_name"`
	DesignationID null.Int    `json:"designation_id" db:"designation_id"`
	Designation   null.String `json:"designation_name" db:"designation"`
	PostStatus    null.String `json:"post_status" db:"post_status"`
	EmployeeID    null.Int    `json:"employee_id" db:"employee_id"`
	EmployeeName  null.String `json:"employee_name" db:"employee_name"`
}

type PostDetailsForRedeployment struct {
	OfficeID                  null.Int    `json:"office_id" db:"office_id"`
	OfficeName                null.String `json:"office_name" db:"office_name"`
	FilledStatus              null.String `json:"filled_status" db:"filled_status"`
	PostStatus                null.String `json:"post_status" db:"post_status"`
	AllowancesAttached        *bool       `json:"allowances_attached" db:"allowances_attached"`
	AllowanceDescription      null.String `json:"allowance_description" db:"allowance_description"`
	UpdatedBy                 null.String `json:"updated_by" db:"updated_by"`
	UpdatedDate               *time.Time  `json:"updated_date" db:"updated_date"`
	Status                    null.String `json:"status" db:"status"`
	Remarks                   null.String `json:"remarks" db:"remarks"`
	ValidFrom                 *time.Time  `json:"valid_from" db:"valid_from"`
	ValidTo                   *time.Time  `json:"valid_to" db:"valid_to"`
	OrderCasemark             null.String `json:"order_casemark" db:"order_casemark"`
	OrderDate                 *time.Time  `json:"order_date" db:"order_date"`
	UploadOrderDocName        null.String `json:"upload_order_doc_name" db:"upload_order_doc_name"`
	EstablishmentRegisterID   null.Int    `json:"establishment_register_id" db:"establishment_register_id"`
	Designation               null.String `json:"designation" db:"designation"`
	PermanentStatus           null.String `json:"permanent_status" db:"permanent_status"`
	EstablishmentRegisterName null.String `json:"establishment_register_name" db:"establishment_register_name"`
	GroupName                 null.String `json:"group_name" db:"group_name"`
	OfficeType                null.String `json:"office_type" db:"office_type"`
	OfficeSupervisor          null.String `json:"office_supervisor" db:"office_supervisor"`
	IsHeadOfTheOffice         *bool       `json:"is_head_of_the_office" db:"is_head_of_the_office"`
	PostID                    null.Int    `json:"post_id" db:"post_id"`
	CadreName                 null.String `json:"cadre_name" db:"cadre_name"`
	GroupID                   null.Int    `json:"group_id" db:"group_id"`
	CadreID                   null.Int    `json:"cadre_id" db:"cadre_id"`
	DesignationID             null.Int    `json:"designation_id" db:"designation_id"`
	PostName                  null.String `json:"post_name" db:"post_name"`
	EmployeeName              null.String `json:"employee_name" db:"employee_name"`
}

type UpdatePostManagementMaster struct {
	ApprovePostID      string    `json:"approve_post_id" db:"approve_post_id"`
	CadreID            int       `json:"cadre_id" db:"cadre_id"`
	CadreName          string    `json:"cadre_name" db:"cadre_name"`
	Designation        string    `json:"designation" db:"designation"`
	DesignationID      int       `json:"designation_id" db:"designation_id"`
	CreatedBy          string    `json:"created_by" db:"created_by"`
	EmployeeGroup      string    `json:"employee_group" db:"employee_group"`
	GradePay           int       `json:"grade_pay" db:"grade_pay"`
	GroupID            int       `json:"group_id" db:"group_id"`
	OfficeID           int       `json:"office_id" db:"office_id"`
	OfficeName         string    `json:"office_name" db:"office_name"`
	OrderCaseMark      string    `json:"order_casemark" db:"order_casemark"`
	OrderDate          time.Time `json:"order_date" db:"order_date"`
	PayLevel           int       `json:"pay_level" db:"pay_level"`
	PostID             int       `json:"post_id" db:"post_id"`
	PostName           string    `json:"post_name" db:"post_name"`
	Remarks            string    `json:"remarks" db:"remarks"`
	Status             string    `json:"status" db:"status"`
	UploadOrderDocName string    `json:"upload_order_doc_name" db:"upload_order_doc_name"`
}

type PostNameDetails struct {
	PostID   null.Int    `json:"post_id" db:"post_id"`
	PostName null.String `json:"post_name" db:"post_name"`
}

type DesignationDetails struct {
	DesignationID null.Int    `json:"designation_id" db:"designation_id"`
	Designation   null.String `json:"designation" db:"designation"`
}

type ExceptionReport struct {
	OfficeID        int    `json:"office_id"`
	GroupName       string `json:"group_name"`
	CadreName       string `json:"cadre_name"`
	PostID          int    `json:"post_id"`
	PostName        string `json:"post_name"`
	CircleName      string `json:"circle_name"`
	RegionName      string `json:"region_name"`
	DivisionName    string `json:"division_name"`
	SubDivisionName string `json:"sub_division_name"`
	OfficeName      string `json:"office_name"`
}

type ExceptionReportOrderCasemarkConsolidated struct {
	OfficeID   int    `json:"office_id"`
	OfficeName string `json:"office_name"`
	PostCount  int    `json:"post_count"`
}

type PostRedeploymentReport struct {
	PostID                   int         `json:"post_id" db:"post_id"`
	CadreName                string      `json:"cadre_name" db:"cadre_name"`
	RedeploymentFromOfficeID int         `json:"redeployment_from_office_id" db:"redeployment_from_office_id"`
	RedeploymentToOfficeID   int         `json:"redeployment_to_office_id" db:"redeployment_to_office_id"`
	RedeploymentOn           time.Time   `json:"redeployment_on" db:"redeployment_on"`
	RedeploymentBy           string      `json:"redeployment_by" db:"redeployment_by"`
	EffectiveFrom            time.Time   `json:"effective_from" db:"effective_from"`
	EffectiveUpto            time.Time   `json:"effective_upto" db:"effective_upto"`
	UploadOrderDocName       null.String `json:"upload_order_doc_name" db:"upload_order_doc_name"`
}

type CadreWiseReports struct {
	CadreID          null.Int `json:"cadre_id" db:"cadre_id"`
	CadreName        string   `json:"cadre_name" db:"cadre_name"`
	TotalPosts       int      `json:"total_posts" db:"total_posts"`
	TotalFilledPosts int      `json:"total_filled_posts" db:"total_filled_posts"`
	TotalVacantPosts int      `json:"total_vacant_posts" db:"total_vacant_posts"`
}

type CadreWiseOfficeWiseReport struct {
	GroupOfficeID    int    `json:"group_office_id" db:"group_office_id"`
	GroupOfficeName  string `json:"group_office_name" db:"group_office_name"`
	OfficeID         int    `json:"office_id" db:"office_id"`
	OfficeName       string `json:"office_name" db:"office_name"`
	CadreID          int    `json:"cadre_id" db:"cadre_id"`
	CadreName        string `json:"cadre_name" db:"cadre_name"`
	TotalPosts       int    `json:"total_posts" db:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts" db:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts" db:"total_vacant_posts"`
}

type ListCadreWiseReport struct {
	PostID        int    `json:"post_id" db:"post_id"`
	PostName      string `json:"post_name" db:"post_name"`
	OfficeId      int    `json:"office_id" db:"office_id"`
	OfficeName    string `json:"office_name" db:"office_name"`
	CadreId       int    `json:"cadre_id" db:"cadre_id"`
	CadreName     string `json:"cadre_name" db:"cadre_name"`
	DesignationID int    `json:"designation_id" db:"designation_id"`
	Designation   string `json:"designation" db:"designation"`
	FilledStatus  string `json:"filled_status" db:"filled_status"`
}

type SanctionedStrengthDetails struct {
	GroupID            int    `json:"group_id" db:"group_id"`
	CadreID            int    `json:"cadre_id" db:"cadre_id"`
	CadreName          string `json:"cadre_name" db:"cadre_name"`
	OfficeID           int    `json:"office_id" db:"office_id"`
	OfficeName         string `json:"office_name" db:"office_name"`
	SanctionedStrength int    `json:"sanctioned_strength"`
}
type Postdetails struct {
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

type PostDetail1 struct {
	CircleName       string `json:"circle_name"`
	CircleOfficeID   int64  `json:"circle_office_id"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

type PostSummaryDetail struct {
	PostManagementID     int64  `json:"postmanagement_id"`
	PostName             string `json:"post_name"`
	Designation          string `json:"designation"`
	FilledStatus         string `json:"filled_status"`
	PostStatus           string `json:"post_status"`
	PayLevel             int64  `json:"pay_level"`
	GradePay             int64  `json:"grade_pay"`
	SanctionedStrength   int64  `json:"sanctioned_strength"`
	PermanentStatus      bool   `json:"permanent_status"`
	AllowancesAttached   bool   `json:"allowances_attached"`
	AllowanceDescription string `json:"allowance_description"`
	GroupName            string `json:"group_name"`
	CadreName            string `json:"cadre_name"`
	PostOfficeName       string `json:"post_office_name"`
	OfficeID             int64  `json:"office_id"`
	OfficeName           string `json:"office_name"`
	OfficeTypeCode       string `json:"office_type_code"`
	Pincode              int64  `json:"pincode"`
	DivisionName         string `json:"division_name"`
	SubdivisionName      string `json:"subdivision_name"`
	CircleName           string `json:"circle_name"`
	RegionName           string `json:"region_name"`
}

type PostSummaryResponse struct {
	Summary []PostDetail1       `json:"summary,omitempty"`
	List    []PostSummaryDetail `json:"list,omitempty"`
	Total   int                 `json:"total"`
}

type CircleSummaryDetail struct {
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

type CircleSummaryResponse struct {
	Summary   []CircleSummaryDetail `json:"summary,omitempty"`
	List      []PostSummaryDetail   `json:"list,omitempty"`
	Hierarchy CircleHierarchy       `json:"hierarchy,omitempty"`
	Total     int                   `json:"total"`
}

type RegionSummaryDetail struct {
	DivisionName     string `json:"division_name"`
	DivisionOfficeID string `json:"division_office_id"`
	RegionName       string `json:"region_name"`
	RegionOfficeID   string `json:"region_office_id"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

type RegionSummaryResponse struct {
	Summary   []RegionSummaryDetail `json:"summary,omitempty"`
	List      []PostSummaryDetail   `json:"list,omitempty"`
	Hierarchy RegionHierarchy       `json:"hierarchy,omitempty"`
	Total     int                   `json:"total"`
}

type DivisionSummaryDetail struct {
	OfficeId         string `json:"office_id"`
	OfficeName       string `json:"office_name"`
	OfficeTypeCode   string `json:"office_type_code"`
	Pincode          string `json:"pincode"`
	Email            string `json:"email"`
	ContactNumber    string `json:"contact_number"`
	DivisionName     string `json:"division_name"`
	DivisionOfficeID string `json:"division_office_id"`
	SubDivisionName  string `json:"subdivision_name"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

type DivisionSummaryResponse struct {
	Summary   []DivisionSummaryDetail `json:"summary,omitempty"`
	List      []PostSummaryDetail     `json:"list,omitempty"`
	Hierarchy DivisionHierarchyInfo   `json:"hierarchy,omitempty"`
	Total     int                     `json:"total"`
}

type PostSummaryDetail1 struct {
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

type PostSummaryResponse1 struct {
	Summary []PostSummaryDetail1 `json:"summary,omitempty"`
	List    []PostSummaryDetail  `json:"list,omitempty"`
	Total   int                  `json:"total"`
}

type DivisionHierarchyInfo struct {
	DivisionName   string
	RegionName     string
	RegionOfficeID int64
	CircleName     string
	CircleOfficeID int64
}

type RegionHierarchy struct {
	RegionName     string
	CircleName     string
	CircleOfficeID int64
}

type CircleHierarchy struct {
	CircleName string
	CircleCode string
}

type PostManagementMasterNew1 struct {
	PostManagementID          null.Int32  `json:"postmanagement_id" db:"postmanagement_id"`
	OfficeID                  null.Int32  `json:"office_id" db:"office_id"`
	PostID                    null.Int32  `json:"post_id" db:"post_id"`
	PostName                  null.String `json:"post_name" db:"post_name"`
	OfficeName                null.String `json:"office_name" db:"office_name"`
	GroupId                   null.Int32  `json:"group_id" db:"group_id"`
	GroupName                 null.String `json:"group_name" db:"group_name"`
	CadreName                 null.String `json:"cadre_name" db:"cadre_name"`
	FilledStatus              null.String `json:"filled_status" db:"filled_status"`
	PostStatus                null.String `json:"post_status" db:"post_status"`
	AllowancesAttached        null.Bool   `json:"allowances_attached" db:"allowances_attached"`
	AllowanceDescription      null.String `json:"allowance_description" db:"allowance_description"`
	CreatedBy                 null.String `json:"created_by" db:"created_by"`
	CreatedOn                 null.Time   `json:"created_on" db:"created_date"`
	ApprovedBy                null.String `json:"approved_by" db:"approved_by"`
	ApprovedOn                null.Time   `json:"approved_on" db:"approved_date"`
	UpdatedBy                 null.String `json:"updated_by" db:"updated_by"`
	UpdatedOn                 null.Time   `json:"updated_on" db:"updated_date"`
	Status                    null.String `json:"status" db:"status"`
	Remarks                   null.String `json:"remarks" db:"remarks"`
	ValidFrom                 null.Time   `json:"valid_from" db:"valid_from"`
	ValidTo                   null.Time   `json:"valid_to" db:"valid_to"`
	OrderCaseMark             null.String `json:"order_casemark" db:"order_casemark"`
	OrderDate                 null.Time   `json:"order_date" db:"order_date"`
	UploadOrderDocName        null.String `json:"upload_order_doc_name" db:"upload_order_doc_name"`
	EstablishmentRegisterID   null.Int32  `json:"establishment_register_id" db:"establishment_register_id"`
	Designation               null.String `json:"designation" db:"designation"`
	PayLevel                  null.Int32  `json:"pay_level" db:"pay_level"`
	GradePay                  null.Int32  `json:"grade_pay" db:"grade_pay"`
	PermanentStatus           null.Bool   `json:"permanent_status" db:"permanent_status"`
	EstablishmentRegisterName null.String `json:"establishment_register_name" db:"establishment_register_name"`
	EmployeeGroup             null.String `json:"employee_group" db:"employee_group"`
	SanctionedStrength        null.Int32  `json:"sanctioned_strength" db:"sanctioned_strength"`
	Count                     null.Int32  `json:"count" db:"count"`
	CadreID                   null.Int32  `json:"cadre_id" db:"cadre_id"`
	DesignationId             null.Int32  `json:"designation_id" db:"designation_id"`
	ApprovePostID             null.String `json:"approve_post_id" db:"approve_post_id"`
	ApproveStatus             null.String `json:"approve_status" db:"approve_status"`
	EmployeeID                null.Int32  `json:"employee_id" db:"employee_id"`     // Employee ID from kafka_employee_master
	EmployeeName              null.String `json:"employee_name" db:"employee_name"` // Employee full name (concatenated)
	OfficeType                null.String `json:"office_type" db:"office_type"`
}

type PostDetails1 struct {
	PostID       null.Int32  `json:"post_id" db:"post_id"`
	PostName     null.String `json:"post_name" db:"post_name"`
	GroupId      null.Int32  `json:"group_id" db:"group_id"`
	Designation  null.String `json:"designation" db:"designation"`
	PostStatus   null.String `json:"post_status" db:"post_status"`
	FilledStatus null.String `json:"filled_status" db:"filled_status"`
}
