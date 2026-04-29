package domain

import (
	"mime/multipart"
	"time"

	"github.com/volatiletech/null/v9"
)

type PosttoPostMap struct {
	EmployeePostID               int       `db:"employee_post_id" json:"employee_post_id"`
	GDSLeaveSancAuthority1       int32     `db:"gds_leave_sanc_authority_1" json:"gds_leave_sanc_authority_1"`
	GDSLeaveSancAuthority2       int32     `db:"gds_leave_sanc_authority_2" json:"gds_leave_sanc_authority_2"`
	ReportingAuthority           int32     `db:"reporting_authority" json:"reporting_authority"`
	AparReportingAuthority       int32     `db:"apar_reporting_authority" json:"apar_reporting_authority"`
	AparReviewAuthority          int32     `db:"apar_review_authority" json:"apar_review_authority"`
	AparAcceptingAuthority       int32     `db:"apar_accepting_authority" json:"apar_accepting_authority"`
	AparRepresentAuthority       int32     `db:"apar_represent_authority" json:"apar_represent_authority"`
	ServiceBookApproveAuthority1 int32     `db:"service_book_approve_authority1" json:"service_book_approve_authority1"`
	LeaveSancAuthority1          int32     `db:"leave_sanc_authority_1" json:"leave_sanc_authority_1"`
	LeaveSancAuthority2          int32     `db:"leave_sanc_authority_2" json:"leave_sanc_authority_2"`
	LeaveSancAuthority3          int32     `db:"leave_sanc_authority_3" json:"leave_sanc_authority_3"`
	UpdatedDate                  time.Time `db:"updated_date" json:"updated_date"`
	UpdatedBy                    string    `db:"updated_by" json:"updated_by"`
	PayApproveAuthority1         int32     `db:"pay_approve_authority1" json:"pay_approve_authority1"`
	//PayApproveAuthority2         int32       `db:"pay_approve_authority2" json:"pay_approve_authority2"`
	LeaveFWDAuthority1           int32       `db:"leave_fwd_authority1" json:"leave_fwd_authority1"`
	LeaveFWDAuthority2           int32       `db:"leave_fwd_authority2" json:"leave_fwd_authority2"`
	PayFWDAuthority1             int32       `db:"pay_fwd_authority1" json:"pay_fwd_authority1"`
	PayFWDAuthority2             int32       `db:"pay_fwd_authority2" json:"pay_fwd_authority2"`
	AppointingAuthority          int32       `db:"appointing_authority" json:"appointing_authority"`
	DisciplinaryAuthority        int32       `db:"disciplinary_authority" json:"disciplinary_authority"`
	DdoAuthority                 int32       `db:"ddo_authority" json:"ddo_authority"`
	AdminAuthority               int32       `db:"admin_authority" json:"admin_authority"`
	PensionSanctioningAuthority  int32       `db:"pension_sanctioning_authority" json:"pension_sanctioning_authority"`
	PensionAuthorisingAuthority  int32       `db:"pension_authorising_authority" json:"pension_authorising_authority"`
	ServiceBookApproveAuthority2 int32       `db:"service_book_approve_authority2" json:"service_book_approve_authority2"`
	RoleAuthority                int32       `db:"role_authority" json:"role_authority"`
	ServiceBookForwardAuthority1 int32       `db:"service_book_foward_authority1" json:"service_book_foward_authority1"`
	ServiceBookForwardAuthority2 int32       `db:"service_book_foward_authority2" json:"service_book_foward_authority2"`
	PostMapID                    string      `db:"post_map_id,pk" json:"post_map_id"`
	PostMapPostId                int32       `json:"post_map_post_id"`
	PostMappingColumnName        string      `json:"post_map_column_name"`
	FieldUpdated                 string      `json:"field_updated"`
	NewValue                     interface{} `json:"new_value"`
	OfficeID                     int         `db:"employee_office_id" json:"employee_office_id"`
	VigilenceMakerAuthority      int32       `db:"vigilence_maker_authority" json:"vigilence_maker_authority"`
	AdminOffice                  int32       `db:"admin_office" json:"admin_office"`
	ApproveStatus                string      `json:"approve_status"`
	OfficeName                   string      `json:"office_name"`
	PostName                     string      `db:"post_name" json:"post_name"`
	AparCustodianAuhtority       int32       `db:"apar_custodian_authority" json:"apar_custodian_authority"`
	CreatedDate                  time.Time   `db:"created_date" json:"created_date"`
	CreatedBy                    string      `db:"created_by" json:"created_by"`
}

// PosttoPostMap represents a mapping record with dynamic fields
type PosttoPostMap1 struct {
	EmployeePostID   int                    `json:"employee_post_id"`
	EmployeeOfficeID int                    `json:"employee_office_id"`
	CreatedDate      time.Time              `json:"created_date"`
	CreatedBy        string                 `json:"created_by"`
	ApproveStatus    string                 `json:"approve_status"`
	FieldName        string                 `json:"field_name"`       // Explicit field_name
	FieldValue       int32                  `json:"field_value"`      // Explicit field_value
	DesignationName  string                 `json:"designation_name"` // Designation name
	OfficeName       string                 `json:"office_name"`      // Office name
	Fields           map[string]interface{} `json:"fields"`
}

//	type PostMapMaster struct {
//		PostID                 int    `json:"post_id"`
//		PostMapID              string `json:"post_map_id"`
//		PostMappingColumnName  string `json:"post_mapping_column_name"`
//		PostMapStatus          string `json:"post_map_status"`
//		PostMapPostId          int    `json:"post_map_post_id"`
//		PostMappingDescription string `json:"post_mapping_description"`
//	}
type PostMapMaster struct {
	PostMapID              string `json:"post_map_id" db:"mapping_id"`
	PostMappingColumnName  string `json:"post_mapping_column_name" db:"post_mapping_column_name"`
	PostMapStatus          string `json:"post_map_status" db:"post_mapping_status"`
	Remarks                string `json:"remarks" db:"remarks"`
	PostID                 int    `json:"post_id" db:"post_mapping_id"`
	PostMappingDescription string `json:"post_mapping_description" db:"post_mapping_description"`
}

// AuthorityDetails represents details about an authority
type AuthorityDetails struct {
	AuthorityName        string
	AuthorityDescription string
	AuthorityPost        int
	DesignationName      string
	OfficeID             int
	OfficeName           string
}

type PosttoPostMapDetail struct {
	EmployeePostID   int         `json:"employee_post_id"`
	EmployeeOfficeID int         `json:"employee_office_id"`
	FieldName        string      `json:"field_name"`
	FieldValue       interface{} `json:"field_value"`
}

type ApprovePostMappingDetail struct {
	EmployeePostID int    `json:"employee_post_id" validate:"required"`
	ApprovedBy     string `json:"approved_by" validate:"required"`
	FieldName      string `json:"field_name" validate:"required"`
	FieldValue     int32  `json:"field_value" validate:"required"`
	Status         string `json:"approve_status" validate:"required"`
	OfficeID       int    `json:"office_id" validate:"required"`
	Remarks        string `json:"remarks"  `
}

type MasterAuthority struct {
	CadreName            string `json:"cadre_name"`
	Designation          string `json:"designation"`
	RoleMappingID        string `json:"role_mapping_id"`
	AuthorityDescription string `json:"authority_description"`
}

//	type PostWithEmployee struct {
//		PostID          int     `json:"post_id"`
//		PostName        string  `json:"post_name"`
//		OfficeID        int     `json:"office_id"`
//		OfficeName      string  `json:"office_name"`
//		GroupID         *int    `json:"group_id"`
//		GroupName       *string `json:"group_name"`
//		CadreID         *int    `json:"cadre_id"`
//		CadreName       *string `json:"cadre_name"`
//		DesignationID   *int    `json:"designation_id"`
//		DesignationName *string `json:"designation_name"`
//		EmployeeID      *int    `json:"employee_id"`
//		EmployeeName    string  `json:"employee_name"`
//	}
type PostMappingChange struct {
	PostID       int            `json:"post_id"`
	OfficeID     int            `json:"office_id"`
	EmployeeID   int            `json:"employee_id"`
	EmployeeName string         `json:"employee_name"`
	Changes      map[string]int `json:"changes"`
}

type PostMappingPayload struct {
	MappingData []PostMappingChange `json:"mapping_data"`
	Meta        struct {
		TotalChanges int           `json:"total_changes"`
		SkippedRows  []int         `json:"skipped_rows"`
		SkippedCells []interface{} `json:"skipped_cells"` // adjust as per actual shape
		ApprovedBy   struct {
			PostID     int `json:"post_id"`
			EmployeeID int `json:"employee_id"`
		} `json:"approved_by"`
		CreatedBy string `json:"created_by"`
		Timestamp string `json:"timestamp"`
		OfficeID  string `json:"office_id"`
	} `json:"meta"`
}

type PostWithEmployee struct {
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

type MinioFile struct {
	FilePath    string
	FileSize    int64
	ContentType string
	File        multipart.File
}
type RedeployedPostAuthority struct {
	AuthorityName   string `json:"authority_name"`
	EmployeeCount   int    `json:"employee_count"`
	EmployeeDetails []EmployeeDetail
}

type EmployeeDetail struct {
	EmployeeName null.String `json:"employee_name"`
	EmployeeID   null.Int    `json:"employee_id"`
	PostID       null.Int    `json:"post_id"`
	PostName     null.String `json:"post_name"`
	OfficeID     null.Int    `json:"office_id"`
	OfficeName   null.String `json:"office_name"`
}

type PostAuthorityDetails struct {
	PostID              null.Int    `json:"post_id"`
	PostName            null.String `json:"post_name"`
	Designation         null.String `json:"designation"`
	CadreID             null.Int    `json:"cadre_id"`
	CadreName           null.String `json:"cadre_name"`
	GroupID             null.Int    `json:"group_id"`
	GroupName           null.String `json:"group_name"`
	EmployeeID          null.Int    `json:"employee_id"`
	EmployeeName        null.String `json:"employee_name"`
	CLSanctionAuthority null.Int    `json:"cl_sanc_authority" db:"leave_sanc_authority_1"`
	ELSanctionAuthority null.Int    `json:"el_sanc_authority" db:"leave_sanc_authority_2"`
	PayApproveAuthority1 null.Int    `json:"pay_approve_authority1" db:"pay_approve_authority1"`
	AppointingAuthority null.Int    `json:"appointing_authority" db:"appointing_authority"`
	DisciplineAuthority null.Int    `json:"discipline_authority" db:"disciplinary_authority"`
	DDOAuthority        null.Int    `json:"ddo_authority" db:"ddo_authority"`
}
