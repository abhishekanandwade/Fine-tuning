-- pmdm.role_authority_mapping definition

-- Drop table

-- DROP TABLE pmdm.role_authority_mapping;

CREATE TABLE pmdm.role_authority_mapping (
	reporting_authority int4 NULL,
	post_name varchar(50) NULL,
	new_office_id int4 NULL,
	office_name varchar(50) NULL,
	group_name varchar(50) NULL,
	cadre varchar(50) NULL,
	designation varchar(50) NULL,
	office_type_code varchar(50) NULL,
	circle_office_id int4 NULL,
	circle_name varchar(50) NULL,
	region_office_id int4 NULL,
	region_name varchar(50) NULL,
	division_office_id int4 NULL,
	division_name varchar(50) NULL,
	sub_division_office_id varchar(50) NULL,
	sub_division_name varchar(50) NULL,
	ho_id int4 NULL,
	ho_name varchar(50) NULL,
	so_id varchar(50) NULL,
	so_name varchar(50) NULL,
	hro_id varchar(50) NULL,
	hro_name varchar(50) NULL,
	sro_id varchar(50) NULL,
	sro_name varchar(50) NULL,
	bo_id varchar(50) NULL,
	bo_name varchar(50) NULL
);