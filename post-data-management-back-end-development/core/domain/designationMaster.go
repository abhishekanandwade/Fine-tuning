package domain

import "time"

// CadreMaster represents a record from the cadre_master table
// type DesignationMaster struct {
// 	DesignationID int       `json:"designation_id"`
// 	Designation   string    `json:"designation"`
// 	GroupName     string    `json:"group_name"`
// 	CadreName     string    `json:"cadre_name"`
// 	CreatedBy     string    `json:"created_by"`
// 	CreatedDate   time.Time `json:"created_date"`
// 	ApprovedBy    string    `json:"approved_by"`
// 	ApprovedDate  time.Time `json:"approved_date"`
// 	ValidFrom     time.Time `json:"valid_from"`
// 	ValidTo       time.Time `json:"valid_to"`
// 	Status        string    `json:"status"`
// 	Remarks       string    `json:"remarks"`
// 	CadreId       int       `json:"cadre_id"`
// 	GroupId       int16     `json:"group_id"`
// }

type DesignationMaster struct {
	DesignationID  int       `json:"designation_id" db:"designation_id" insert_designation_master:"designation_id" update_designation_master:"designation_id"`
	Designation    string    `json:"designation" db:"designation" insert_designation_master:"designation" update_designation_master:"designation"`
	GroupName      string    `json:"group_name" db:"group_name" insert_designation_master:"group_name" update_designation_master:"group_name"`
	CadreName      string    `json:"cadre_name" db:"cadre_name" insert_designation_master:"cadre_name" update_designation_master:"cadre_name"`
	CreatedBy      string    `json:"created_by" db:"created_by" insert_designation_master:"created_by"`
	CreatedDate    time.Time `json:"created_date" db:"created_date" insert_designation_master:"created_date"`
	ApprovedBy     string    `json:"approved_by" db:"approved_by"`
	ApprovedDate   time.Time `json:"approved_date" db:"approved_date"`
	ValidFrom      time.Time `json:"valid_from" db:"valid_from" insert_designation_master:"valid_from" update_designation_master:"valid_from"`
	ValidTo        time.Time `json:"valid_to" db:"valid_to" insert_designation_master:"valid_to" update_designation_master:"valid_to"`
	Status         string    `json:"status" db:"status" insert_designation_master:"status" update_designation_master:"status"`
	Remarks        string    `json:"remarks" db:"remarks" insert_designation_master:"remarks" update_designation_master:"remarks"`
	CadreId        int       `json:"cadre_id" db:"cadre_id" insert_designation_master:"cadre_id" update_designation_master:"cadre_id"`
	GroupId        int16     `json:"group_id" db:"group_id" insert_designation_master:"group_id" update_designation_master:"group_id"`
	DesignationUID int       `json:"designation_uid" db:"designation_uid"`
}
type DesignationMasterNew struct {
	DesignationID int    `json:"designation_id"`
	Designation   string `json:"designation"`
	CadreID       int    `json:"cadre_id"`
	CadreName     string `json:"cadre_name"`
}
