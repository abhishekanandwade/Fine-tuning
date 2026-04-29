-- pmdm.cadre_master definition

-- Drop table

-- DROP TABLE pmdm.cadre_master;

CREATE TABLE pmdm.cadre_master (
	cadre_id int4 DEFAULT nextval('pmdm.cadre_mastser_cadre_code_seq'::regclass) NOT NULL,
	cadre_name varchar(50) NULL,
	group_name varchar(30) NULL,
	pay_level int4 NULL,
	grade_pay int4 NULL,
	created_by varchar(30) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	updated_by varchar(30) NULL,
	updated_date timestamp NULL,
	valid_from timestamp NULL,
	valid_to timestamp NULL,
	status varchar(50) NULL,
	remarks varchar(200) NULL,
	group_id int4 NULL,
	CONSTRAINT cadre_mastser_pk PRIMARY KEY (cadre_id)
);