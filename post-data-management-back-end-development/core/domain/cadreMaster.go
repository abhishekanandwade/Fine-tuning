package domain

import (
	"time"

	"github.com/volatiletech/null/v9"
)

// CadreMaster represents a record from the cadre_master table
type CadreMaster struct {
	CadreID     null.Int32  `json:"cadre_id"`
	CadreName   null.String `json:"cadre_name" insert_cadre_master:"cadre_name" update_cadre_master:"cadre_name"`
	GroupName   null.String `json:"group_name" insert_cadre_master:"group_name" update_cadre_master:"group_name"`
	Designation null.String `json:"designation" insert_cadre_master:"designation" update_cadre_master:"designation"`
	PayLevel    null.Int32  `json:"pay_level" insert_cadre_master:"pay_level" update_cadre_master:"pay_level"`
	GradePay    null.Int32  `json:"grade_pay" insert_cadre_master:"grade_pay" update_cadre_master:"grade_pay"`
	CreatedBy   null.String `json:"created_by" insert_cadre_master:"created_by"`
	CreatedOn   null.Time   `json:"created_on"`
	ApprovedBy  null.String `json:"approved_by" `
	ApprovedOn  null.Time   `json:"approved_on" `
	UpdatedBy   null.String `json:"updated_by" update_cadre_master:"updated_by"`
	UpdatedOn   null.Time   `json:"updated_on"`
	ValidFrom   null.Time   `json:"valid_from" insert_cadre_master:"valid_from" update_cadre_master:"valid_from"`
	ValidTo     null.Time   `json:"valid_to" insert_cadre_master:"valid_to" update_cadre_master:"valid_to"`
	Status      null.String `json:"status" insert_cadre_master:"status" update_cadre_master:"status"`
	Remarks     null.String `json:"remarks" insert_cadre_master:"remarks" update_cadre_master:"remarks"`
	GroupCode   null.Int16  `json:"group_id" insert_cadre_master:"group_id" update_cadre_master:"group_id"`
}
type CadreMasterNew struct {
	CadreID   int    `json:"cadre_id"`
	CadreName string `json:"cadre_name"`
}

type Summary struct {
	GroupName   string `json:"group_name"`
	CadreName   string `json:"cadre_name"`
	TotalPosts  int    `json:"total_posts"`
	TotalFilled int    `json:"total_filled_posts"`
	TotalVacant int    `json:"total_vacant_posts"`
}

type Detail struct {
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

type CadreInfo struct {
	CadreID   int64  `json:"cadre_id"`
	CadreName string `json:"cadre_name"`
	GroupName string `json:"group_name"`
}

type CountSummary struct {
	TotalPosts  int `json:"total_posts"`
	TotalFilled int `json:"total_filled_posts"`
	TotalVacant int `json:"total_vacant_posts"`
}

type CircleSummary struct {
	CircleName       string `json:"circle_name"`
	CircleOfficeID   int    `json:"circle_office_id"`
	GroupName        string `json:"group_name"`
	CadreName        string `json:"cadre_name"`
	TotalPosts       int    `json:"total_posts"`
	TotalFilledPosts int    `json:"total_filled_posts"`
	TotalVacantPosts int    `json:"total_vacant_posts"`
}

type DetailedPost struct {
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

type HierarchyInfo struct {
	Level     int64  `json:"level"`
	LevelName string `json:"level_name"`
	CadreName string `json:"cadre_name"`
}

type DivisionSummary struct {
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

type DivisionDetail struct {
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

type RegionInfo struct {
	Level          int64  `json:"level"`
	LevelName      string `json:"level_name"`
	CadreName      string `json:"cadre_name"`
	RegionOfficeId int64  `json:"region_office_id"`
	RegionName     string `json:"region_name"`
	CircleName     string `json:"circle_name"`
	CircleOfficeID int    `json:"circle_office_id"`
}

type HierarchySummary struct {
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

type HierarchyDetail struct {
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

type HierarchyInfodata struct {
	ParentOfficeID   int64  `json:"parent_office_id"`
	ParentOfficeName string `json:"parent_office_name"`
	CadreName        string `json:"cadre_name"`
	Level            int    `json:"level"`
	OfficeTypeCode   string `json:"office_type_code"`
}

type OfficeData struct {
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

type OfficePostDetail struct {
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

type OfficeInfo struct {
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

type PostDetail struct {
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

type ContextInfo struct {
	OfficeName     string `json:"office_name"`
	OfficeTypeCode string `json:"office_type_code"`
	DivisionName   string `json:"division_name"`
	RegionName     string `json:"region_name"`
	CircleName     string `json:"circle_name"`
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

type PostSummary struct {
	TotalPosts  int `json:"totalPosts"`
	TotalFilled int `json:"totalFilled"`
	TotalVacant int `json:"totalVacant"`
}

type HierarchyPost struct {
	Level            int    `json:"level"`
	LevelName        string `json:"level_name"`
	CadreID          int64  `json:"cadre_id"`
	OfficeID         *int64 `json:"office_id,omitempty"`
	DivisionOfficeID *int64 `json:"division_office_id,omitempty"`
	RegionOfficeID   *int64 `json:"region_office_id,omitempty"`
	CircleOfficeID   *int64 `json:"circle_office_id,omitempty"`
}

type RegionRequest struct {
	CadreName      string `form:"cadre_name" binding:"required"`
	CircleOfficeID int64  `form:"circle_office_id" binding:"required"`
	RegionOfficeID int64  `form:"region_office_id"`
	Search         string `form:"search"`
	IncludeList    bool   `form:"includeList"`
}

type RegionSummary struct {
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

type RegionDetail struct {
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

type CircleInfo struct {
	CircleName string `json:"circle_name"`
	CircleCode string `json:"circle_code"`
}

type HierarchyRegion struct {
	Level          int    `json:"level"`
	LevelName      string `json:"level_name"`
	CadreName      string `json:"cadre_name"`
	CircleOfficeID int64  `json:"circle_office_id"`
	CircleName     string `json:"circle_name"`
}

type CadreMasterD1 struct {
	CadreID   int       `json:"cadre_id" db:"cadre_id"`
	CadreName string    `json:"cadre_name" db:"cadre_name"`
	GroupName string    `json:"group_name" db:"group_name"`
	PayLevel  int       `json:"pay_level" db:"pay_level"`
	GradePay  int       `json:"grade_pay" db:"grade_pay"`
	ValidFrom time.Time `json:"valid_from" db:"valid_from"`
	ValidTo   time.Time `json:"valid_to" db:"valid_to"`
	Status    string    `json:"status" db:"status"`
	Remarks   string    `json:"remarks" db:"remarks"`
	GroupID   int       `json:"group_id" db:"group_id"`
}
