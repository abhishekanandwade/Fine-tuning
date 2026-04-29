package bootstrap

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"time"

	"github.com/go-playground/validator/v10"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/rs/zerolog"
	auth "gitlab.cept.gov.in/it-2.0-common/api-authz"
	config "gitlab.cept.gov.in/it-2.0-common/api-config"
	db "gitlab.cept.gov.in/it-2.0-common/api-db"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"

	//validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
	handler "pmdm/handler"
	repo "pmdm/repo/postgres"

	"go.uber.org/fx"
)

var Fxconfig = fx.Module(
	"configmodule",
	fx.Provide(
		config.NewDefaultConfigFactory,
		newFxConfig,
	),
)

type FxConfigParam struct {
	fx.In
	Factory config.ConfigFactory
}

func newFxConfig(p FxConfigParam) (*config.Config, error) {
	return p.Factory.Create(
		config.WithFileName("config"),
		config.WithAppEnv(os.Getenv("APP_ENV")),
		config.WithFilePaths(
			".",
			"./configs",
			os.Getenv("APP_CONFIG_PATH"),
		),
	)
}

var Fxlog = fx.Module(
	"logmodule",
	fx.Provide(
		log.NewDefaultLoggerFactory,
	),
	fx.Invoke(newFxLogger),
)

// type FxMinioParam struct {
// 	fx.In
// 	Factory log.LoggerFactory
// 	Config  *config.Config
// }

// func newFxMinio(p FxMinioParam) {
// 	var err error
// 	var MinioClient *minio.Client

// 	MinioClient, err = minio.New(p.Config.GetString("minio.url"), &minio.Options{
// 		Creds:  credentials.NewStaticV4(p.Config.GetString("minio.AccessKey"), p.Config.GetString("minio.SecretKey"), ""),
// 		Secure: true})
// 	if err != nil {
// 		log.GetBaseLoggerInstance().ToZerolog().Error().Msg("Minio Client Error")
// 	}

// 	exists, errBucketExists := MinioClient.BucketExists(context.Background(), p.Config.GetString("minio.BucketName"))

// 	if errBucketExists != nil {
// 		log.GetBaseLoggerInstance().ToZerolog().Error().Msg("Error checking if bucket exists:")
// 	}

// 	if exists {
// 		log.GetBaseLoggerInstance().ToZerolog().Debug().Msg("Bucket found")
// 	} else {
// 		log.GetBaseLoggerInstance().ToZerolog().Error().Msg("Bucket does not exist")

// 	}

// }

// var FxMinIO = fx.Module(
//
//	"MinIOModule",
//	// fx.Provide(
//	// 	log.NewDefaultLoggerFactory,
//	// ),
//	fx.Provide(func(p FxMinioParam) (*minio.Client, error) {
//		return minio.New(p.Config.GetString("minio.url"), &minio.Options{
//			Creds:  credentials.NewStaticV4(p.Config.GetString("minio.AccessKey"), p.Config.GetString("minio.SecretKey"), ""),
//			Secure: true,
//		})
//	}),
//	fx.Invoke(newFxMinio),
//
// )
type FxLogParam struct {
	fx.In
	Factory log.LoggerFactory
	Config  *config.Config
}

func newFxLogger(p FxLogParam) error {

	var level zerolog.Level
	if p.Config.AppDebug() {
		level = zerolog.DebugLevel
	} else {
		level = log.FetchLogLevel(p.Config.GetString("log.level"))
	}

	err := p.Factory.Create(
		log.WithServiceName(p.Config.AppName()),
		log.WithLevel(level),
		log.WithOutputWriter(os.Stdout),
	)
	if err != nil {
		return err
	}

	return nil
}

// type FxDBParam struct {
// 	fx.In
// 	//Factory db.DBFactory
// 	Config *config.Config
// 	log    *log.Logger
// 	db     *db.DB
// 	lc     fx.Lifecycle
// }

func dbconfig(c *config.Config) db.DBConfig {
	dbconfig := db.DBConfig{

		DBUsername:        c.GetString("db.username"),
		DBPassword:        c.GetString("db.password"),
		DBHost:            c.GetString("db.host"),
		DBPort:            c.GetString("db.port"),
		DBDatabase:        c.GetString("db.database"),
		Schema:            c.GetString("db.schema"),
		MaxConns:          c.GetInt32("db.maxconns"),
		MinConns:          c.GetInt32("db.minconns"),
		MaxConnLifetime:   time.Duration(c.GetInt("db.maxconnlifetime")),
		MaxConnIdleTime:   time.Duration(c.GetInt("db.maxconnidletime")),
		HealthCheckPeriod: time.Duration(c.GetInt("db.healthcheckperiod")),
		AppName:           c.AppName(),
	}
	return dbconfig

}

var FxDB = fx.Module(
	"DBModule",
	fx.Provide(
		dbconfig,
		//dbconfig_old,
		db.NewDefaultDbFactory().NewPreparedDBConfig,
		db.NewDefaultDbFactory().CreateConnection,
	),
	fx.Invoke(dblifecycle),
)

// func newfxDB(Config *config.Config) {
// 	db.NewDefaultDbFactory().NewPreparedDBConfig( Config),

// 	//f.Create(Config)

// }

func dblifecycle(db *db.DB, lc fx.Lifecycle) {
	lc.Append(
		fx.Hook{
			OnStart: func(ctx context.Context) error {

				log.GetBaseLoggerInstance().ToZerolog().Info().Str("module", "DBModule").Msg("Starting fxdb module")
				err := db.Ping()
				if err != nil {
					return err
				}
				log.GetBaseLoggerInstance().ToZerolog().Info().Msg("Successfully connected to the database")

				return nil
			},
			OnStop: func(ctx context.Context) error {
				db.Close()
				return nil
			},
		},
	)
}

// NewValidatorService add it as part of fx invoke
var Fxvalidator = fx.Module(
	"validator",
	fx.Invoke(handler.NewValidatorService),
)

func HourValidate(f1 validator.FieldLevel) bool {
	timeRegex := regexp.MustCompile(`^([01]\d|2[0-3]):([0-5]\d)$`)
	return timeRegex.MatchString(f1.Field().String())
}

var Fxclient = fx.Module(
	"client",
	fx.Invoke(client),
)

func client() error {

	defaultTimeout := 10 * time.Second
	defaultMaxRetries := 3
	defaultRetryWait := 500 * time.Millisecond
	defaultMaxRetryWait := 3 * time.Second
	err := auth.Init(
		auth.ClientConfig{
			Timeout:      defaultTimeout,
			RetryWait:    defaultRetryWait,
			MaxRetryWait: defaultMaxRetryWait,
			MaxRetries:   defaultMaxRetries,
		},
	)
	if err != nil {
		return err
	}

	return nil

}

// var Fxrouter = fx.Module(
// 	"router",
// 	fx.Provide(router.Defaultgin),
// 	fx.Invoke(router.Routes, startServer),
// )

// func startServer(lc fx.Lifecycle, sv *router.Router) {
// 	eg, _ := errgroup.WithContext(context.Background())

// 	lc.Append(fx.Hook{
// 		OnStop: func(ctx context.Context) error {
// 			ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
// 			defer cancel()
// 			return sv.Shutdown(ctx)
// 		},
// 	})

// 	eg.Go(func() error {
// 		if err := sv.Start(); err != nil && err != http.ErrServerClosed {
// 			return err
// 		}
// 		return nil
// 	})

// }

var FxRepo = fx.Module(
	"Repomodule",
	fx.Provide(
		repo.NewPosttoPostMappingRepository,
		repo.NewCadreMasterRepository,
		repo.NewPostManagementRepository,
		repo.NewDesignationMasterRepository,
	),
)

var FxHandler = fx.Module(
	"Handlermodule",
	fx.Provide(
		handler.NewPosttoPostMappingrHandler,
		handler.NewCadreMasterHandler,
		handler.NewPostManagementHandler,
		handler.NewDesignationMasterHandler,
	),
)

// func dbconfig_old(c *config.Config) db.DBConfig {
// 	dbconfig := db.DBConfig{

// 		DBUsername:        c.DBUsername(),
// 		DBPassword:        c.DBPassword(),
// 		DBHost:            c.DBHost(),
// 		DBPort:            c.DBPort(),
// 		DBDatabase:        c.DBDatabase(),
// 		Schema:            c.DBSchema(),
// 		MaxConns:          int32(c.MaxConns()),
// 		MinConns:          int32(c.MinConns()),
// 		MaxConnLifetime:   time.Duration(c.MaxConnLifetime()),
// 		MaxConnIdleTime:   time.Duration(c.MaxConnIdleTime()),
// 		HealthCheckPeriod: time.Duration(c.HealthCheckPeriod()),
// 		AppName:           c.AppName(),
// 	}
// 	return dbconfig

// }

func InitMinio(c *config.Config) *minio.Client {
	var err error
	MinioClient, err := minio.New(c.GetString("minio.url"), &minio.Options{
		Creds:  credentials.NewStaticV4(c.GetString("minio.accessKey"), c.GetString("minio.secretKey"), ""),
		Secure: true,
	})
	if err != nil {
		log.Fatal(nil, err)
	}

	exists, errBucketExists := MinioClient.BucketExists(context.Background(), c.GetString("minio.bucketName"))

	if errBucketExists != nil {
		log.Fatal(nil, "Error checking if bucket exists: %v", errBucketExists)
	}

	if exists {
		fmt.Println("Bucket found")
	} else {
		log.Fatal(nil, "Bucket %s does not exist", c.GetString("minio.bucketName"))
	}
	return MinioClient
}

var FxMinio = fx.Module(
	"Miniomodule",
	fx.Provide(InitMinio),
)
