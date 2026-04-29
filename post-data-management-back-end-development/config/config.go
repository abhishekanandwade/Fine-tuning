package config

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/spf13/viper"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
)

type MinioConfig struct {
	URL        string `mapstructure:"url"`
	AccessKey  string `mapstructure:"accessKey"`
	SecretKey  string `mapstructure:"secretKey"`
	BucketName string `mapstructure:"bucketName"`
}

var (
	once        sync.Once
	instance    Econfig
	MinioClient *minio.Client
	minioConfig MinioConfig
)

type config struct {
	AppName            string
	AppEnv             string
	DBConnection       string
	TokenSymmetricKey  string
	HttpUrl            string
	HttpPort           string
	DBHost             string
	DBPort             string
	DBdatabase         string
	DBUsername         string
	DBPassword         string
	TokenDuration      string
	RedisServer        string
	RedisPassword      string
	HttpAllowedOrigins string
	Loglevel           string
	ShutDownTime       string
	ShutDowntype       string

	MaxConns           int
	MinConns           int
	MaxConnLifetime    int
	MaxConnIdleTime    int
	HealthCheckPeriod  int
	HealthCheckTimeout int

	AuthDefaultTimeout      string
	AuthDefaultMaxRetries   int
	AuthDefaultRetryWait    string
	AuthDefaultMaxRetryWait string

	DBQueryTimeoutLow       time.Duration
    DBQueryTimeoutMed       time.Duration
}

/*
type Econfig struct {
	appName            string `yaml:"AppName"`
	appEnv             string `yaml:"AppEnv"`
	dBConnection       string `yaml:"DBConnection"`
	tokenSymmetricKey  string `yaml:"TokenSymmetricKey"`
	httpUrl            string `yaml:"HttpUrl"`
	httpPort           string `yaml:"HttpPort"`
	dBHost             string `yaml:"DBHost"`
	dBPort             string `yaml:"DBPort"`
	dBdatabase         string `yaml:"DBdatabase"`
	dBUsername         string `yaml:"DBUsername"`
	dBPassword         string `yaml:"DBPassword"`
	tokenDuration      string `yaml:"TokenDuration"`
	redisServer        string `yaml:"RedisServer"`
	redisPassword      string `yaml:"RedisPassword"`
	httpAllowedOrigins string `yaml:"HttpAllowedOrigins"`
	loglevel           string `yaml:"Loglevel"`
	shutDownTime       string `yaml:"ShutDownTime"`
	shutDowntype       string `yaml:"ShutDowntype"`
}*/

type Econfig struct {
	appName            string `mapstructure:"AppName"`
	appEnv             string `mapstructure:"AppEnv"`
	dBConnection       string `mapstructure:"DBConnection"`
	tokenSymmetricKey  string `mapstructure:"TokenSymmetricKey"`
	httpUrl            string `mapstructure:"HttpUrl"`
	httpPort           string `mapstructure:"HttpPort"`
	dBHost             string `mapstructure:"DBHost"`
	dBPort             string `mapstructure:"DBPort"`
	dBdatabase         string `mapstructure:"DBdatabase"`
	dBUsername         string `mapstructure:"DBUsername"`
	dBPassword         string `mapstructure:"DBPassword"`
	tokenDuration      string `mapstructure:"TokenDuration"`
	redisServer        string `mapstructure:"RedisServer"`
	redisPassword      string `mapstructure:"RedisPassword"`
	httpAllowedOrigins string `mapstructure:"HttpAllowedOrigins"`
	loglevel           string `mapstructure:"Loglevel"`
	shutDownTime       string `mapstructure:"ShutDownTime"`
	shutDowntype       string `mapstructure:"ShutDowntype"`
	maxConns           int    `mapstructure:"MaxConns"`
	minConns           int    `mapstructure:"MinConns"`
	maxConnLifetime    int    `mapstructure:"MaxConnLifetime"`
	maxConnIdleTime    int    `mapstructure:"MaxConnIdleTime"`
	healthcheckPeriod  int    `mapstructure:"HealthCheckPeriod"`
	healthcheckTimeout int    `mapstructure:"HealthCheckTimeout"`

	authDefaultTimeout      string `mapstructure:"AuthDefaultTimeout"`
	authDefaultMaxRetries   int    `mapstructure:"AuthDefaultMaxRetries"`
	authDefaultRetryWait    string `mapstructure:"AuthDefaultRetryWait"`
	authDefaultMaxRetryWait string `mapstructure:"AuthDefaultMaxRetryWait"`

	dbQueryTimeoutLow       time.Duration `mapstructure:"DBQueryTimeoutLow"`
    dbQueryTimeoutMed       time.Duration `mapstructure:"DBQueryTimeoutMed"`
}

func NewConfig(c config) Econfig {
	return Econfig{
		appName:            c.AppName,
		appEnv:             c.AppEnv,
		dBConnection:       c.DBConnection,
		tokenSymmetricKey:  c.TokenSymmetricKey,
		httpUrl:            c.HttpUrl,
		httpPort:           c.HttpPort,
		dBHost:             c.DBHost,
		dBPort:             c.DBPort,
		dBdatabase:         c.DBdatabase,
		dBUsername:         c.DBUsername,
		dBPassword:         c.DBPassword,
		tokenDuration:      c.TokenDuration,
		redisServer:        c.RedisServer,
		redisPassword:      c.RedisPassword,
		httpAllowedOrigins: c.HttpAllowedOrigins,
		loglevel:           c.Loglevel,
		shutDownTime:       c.ShutDownTime,
		shutDowntype:       c.ShutDowntype,
		maxConns:           c.MaxConns,
		minConns:           c.MinConns,
		maxConnLifetime:    c.MaxConnLifetime,
		maxConnIdleTime:    c.MaxConnIdleTime,
		healthcheckPeriod:  c.HealthCheckPeriod,
		healthcheckTimeout: c.HealthCheckTimeout,

		authDefaultTimeout:      c.AuthDefaultTimeout,
		authDefaultMaxRetries:   c.AuthDefaultMaxRetries,
		authDefaultRetryWait:    c.AuthDefaultRetryWait,
		authDefaultMaxRetryWait: c.AuthDefaultMaxRetryWait,

		dbQueryTimeoutLow:       c.DBQueryTimeoutLow,
        dbQueryTimeoutMed:       c.DBQueryTimeoutMed,
	}
}
func InitMinio() {
	var err error
	MinioClient, err = minio.New(minioConfig.URL, &minio.Options{
		Creds:  credentials.NewStaticV4(minioConfig.AccessKey, minioConfig.SecretKey, ""),
		Secure: true,
	})
	if err != nil {
		log.Fatal(nil,err)
	}

	exists, errBucketExists := MinioClient.BucketExists(context.Background(), minioConfig.BucketName)

	if errBucketExists != nil {
		log.Fatal(nil,"Error checking if bucket exists: %v", errBucketExists)
	}

	if exists {
		fmt.Println("Bucket found")
	} else {
		log.Fatal(nil,"Bucket %s does not exist", minioConfig.BucketName)
	}
}

func GetBucketName() string {
	return minioConfig.BucketName
}

func Load() Econfig {
	c := config{}
	once.Do(func() {
		viper.SetConfigName("config")
		viper.AddConfigPath(".")
		viper.SetConfigType("yaml")

		//log := logger.New("DEBUG")

		// Attempt to read the configuration file
		if err := viper.ReadInConfig(); err != nil {
			log.Error(nil,"Failed to read configuration", err)
			os.Exit(1)
		}

		// Unmarshal the configuration into the struct
		if err := viper.Unmarshal(&c); err != nil {
			log.Error(nil,"Failed to unmarshal configuration", err)
			os.Exit(1)
		}

		instance = NewConfig(c)
	})
	return instance
}
func InitConfig() {
	viper.SetConfigName("config")
	viper.SetConfigType("yml")
	viper.AddConfigPath(".")

	if err := viper.ReadInConfig(); err != nil {
		log.Fatal(nil,"Error reading config file, %s", err)
	}

	if err := viper.Sub("minio").Unmarshal(&minioConfig); err != nil {
		log.Fatal(nil,"Unable to decode into struct, %v", err)
	}
}

func (c *Econfig) AppName() string {
	return c.appName

}

func (c *Econfig) AppEnv() string {
	return c.appEnv

}
func (c *Econfig) DBConnection() string {
	return c.dBConnection

}

func (c *Econfig) TokenSymmetricKey() string {
	return c.tokenSymmetricKey

}

func (c *Econfig) Dbhost() string {
	return c.dBHost

}

// HttpUrl returns the httpUrl field value.
func (c *Econfig) HttpUrl() string {
	return c.httpUrl
}

// HttpPort returns the httpPort field value.
func (c *Econfig) HttpPort() string {
	return c.httpPort
}

// DBHost returns the dBHost field value.
func (c *Econfig) DBHost() string {
	return c.dBHost
}

// DBPort returns the dBPort field value.
func (c *Econfig) DBPort() string {
	return c.dBPort
}

// DBDatabase returns the dBdatabase field value.
func (c *Econfig) DBDatabase() string {
	return c.dBdatabase
}

// DBUsername returns the dBUsername field value.
func (c *Econfig) DBUsername() string {
	return c.dBUsername
}

// DBPassword returns the dBPassword field value.
func (c *Econfig) DBPassword() string {
	return c.dBPassword
}

// TokenDuration returns the tokenDuration field value.
func (c *Econfig) TokenDuration() string {
	return c.tokenDuration
}

// RedisServer returns the redisServer field value.
func (c *Econfig) RedisServer() string {
	return c.redisServer
}

// RedisPassword returns the redisPassword field value.
func (c *Econfig) RedisPassword() string {
	return c.redisPassword
}

// HttpAllowedOrigins returns the httpAllowedOrigins field value.
func (c *Econfig) HttpAllowedOrigins() string {
	return c.httpAllowedOrigins
}

// LogLevel returns the loglevel field value.
func (c *Econfig) LogLevel() string {
	return c.loglevel
}

// ShutDownTime returns the shutDownTime field value.
func (c *Econfig) ShutDownTime() string {
	return c.shutDownTime
}

// ShutDownType returns the shutDowntype field value.
func (c *Econfig) ShutDownType() string {
	return c.shutDowntype
}

func (c *Econfig) MaxConns() int {
	return c.maxConns
}

func (c *Econfig) MinConns() int {
	return c.minConns
}

func (c *Econfig) MaxConnLifetime() int {
	return c.maxConnLifetime
}

func (c *Econfig) MaxConnIdleTime() int {
	return c.maxConnIdleTime
}
func (c *Econfig) HealthCheckPeriod() int {
	return c.healthcheckPeriod
}
func (c *Econfig) HealthCheckTimeout() int {
	return c.healthcheckTimeout
}

func (c *Econfig) GetAuthDefaultTimeout() string {
	return c.authDefaultTimeout
}

func (c *Econfig) GetAuthDefaultMaxRetries() int {
	return c.authDefaultMaxRetries
}

func (c *Econfig) GetAuthDefaultRetryWait() string {
	return c.authDefaultRetryWait
}

func (c *Econfig) GetAuthDefaultMaxRetryWait() string {
	return c.authDefaultMaxRetryWait
}

func (c *Econfig) GetDBQueryTimeoutLow() time.Duration {
    return c.dbQueryTimeoutLow
}

func (c *Econfig) GetDBQueryTimeoutMed() time.Duration {
    return c.dbQueryTimeoutMed
}
