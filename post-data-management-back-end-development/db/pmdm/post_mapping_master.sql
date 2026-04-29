-- pmdm.post_mapping_master definition

-- Drop table

-- DROP TABLE pmdm.post_mapping_master;

CREATE TABLE pmdm.post_mapping_master (
	mapping_id varchar(30) NULL,
	post_mapping_column_name varchar(100) NULL,
	post_mapping_status varchar(30) NULL,
	remarks varchar(200) NULL,
	post_mapping_id int4 NOT NULL,
	post_mapping_description varchar(200) NULL,
	CONSTRAINT post_mapping_master_pk PRIMARY KEY (post_mapping_id),
	CONSTRAINT post_mapping_master_un UNIQUE (mapping_id)
);