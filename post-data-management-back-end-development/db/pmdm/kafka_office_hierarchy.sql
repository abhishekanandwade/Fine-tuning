-- pmdm.kafka_office_hierarchy definition

-- Drop table

-- DROP TABLE pmdm.kafka_office_hierarchy;

CREATE TABLE pmdm.kafka_office_hierarchy (
	office_hierarchy_id int4 DEFAULT 0 NOT NULL,
	office_id int4 NOT NULL,
	office_type_id int4 NULL,
	office_type_code varchar(20) NULL,
	circle_name varchar(50) NULL,
	circle_code varchar(20) NULL,
	circle_office_id int4 NULL,
	region_name varchar(50) NULL,
	region_office_id int4 NULL,
	division_name varchar(50) NULL,
	division_office_id int4 NULL,
	subdivision_name varchar(50) NULL,
	subdivision_office_id int4 NULL,
	ho_id int4 NULL,
	ho_name varchar(50) NULL,
	hro_id int4 NULL,
	hro_name varchar(50) NULL,
	so_id int4 NULL,
	so_name varchar(50) NULL,
	sro_id int4 NULL,
	sro_name varchar(50) NULL,
	accounting_office_id int4 NULL,
	bo_id int4 DEFAULT 0 NULL,
	bo_name varchar(50) DEFAULT ''::text NULL,
	sub_division_name text DEFAULT ''::text NULL,
	sub_division_office_id int4 DEFAULT 0 NULL,
	CONSTRAINT kafka_office_hierarchy_pk PRIMARY KEY (office_id)
);