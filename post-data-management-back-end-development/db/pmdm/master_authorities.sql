-- pmdm.master_authorities definition

-- Drop table

-- DROP TABLE pmdm.master_authorities;

CREATE TABLE pmdm.master_authorities (
	office_id int4 NULL,
	office_type varchar(50) NULL,
	office_name varchar(50) NULL,
	cadre_name varchar(50) NULL,
	designation varchar(50) NULL,
	post_id int4 NULL,
	emp_id int4 NULL,
	role_mapping_id varchar(50) NULL,
	authority_description varchar(50) NULL,
	division_office_id int4 NULL,
	region_office_id int4 NULL,
	circle_office_id int4 NULL,
	admin_office_id int4 NULL
);