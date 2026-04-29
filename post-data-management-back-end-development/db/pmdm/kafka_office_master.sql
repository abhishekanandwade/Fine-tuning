-- pmdm.kafka_office_master definition

-- Drop table

-- DROP TABLE pmdm.kafka_office_master;

CREATE TABLE pmdm.kafka_office_master (
	office_id int4 NOT NULL,
	office_name varchar(50) NULL,
	office_type_id int4 NULL,
	office_type_code varchar(20) NULL,
	email_id varchar(50) NULL,
	contact_number varchar(20) NULL,
	office_class varchar(50) NULL,
	pincode int4 NULL,
	reporting_office_id int4 NULL,
	office_status_id int4 NULL,
	csi_facility_id varchar(20) NULL,
	sol_id varchar(20) NULL,
	ddo_code varchar(20) NULL,
	office_level varchar(20) NULL,
	working_days varchar(100) NULL,
	accounting_office_id int4 DEFAULT 0 NULL,
	delivery_office_flag bool NULL,
	pli_id varchar(20) NULL,
	remarks varchar(200) NULL,
	valid_from date NULL,
	valid_to date NULL,
	CONSTRAINT kafka_office_master_pkey PRIMARY KEY (office_id)
);