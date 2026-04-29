CREATE TABLE pmdm.designation_master (
	designation_id serial4 NOT NULL,
	designation varchar(50) NULL,
	group_name varchar(30) NULL,
	cadre_name varchar(50) NULL,
	created_by varchar(30) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	valid_from timestamp NULL,
	valid_to timestamp NULL,
	status varchar(30) NULL,
	remarks varchar(200) NULL,
	cadre_id int4 NULL,
	group_id int4 NULL,
	CONSTRAINT designation_master_pk PRIMARY KEY (designation_id)
);