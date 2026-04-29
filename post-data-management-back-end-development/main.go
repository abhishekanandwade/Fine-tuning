// Swagger
//
//	@title                       Post Management Data Management API
//	@version                     1.0
//	@description                 A comprehensive API for managing Post Management Data Management, offering endpoints for creation, update, deletion, and retrieval of Posts Data.
//	@termsOfService              http://cept.gov.in/terms
//	@contact.name                API Support Team
//	@contact.url                 http://cept.gov.in/support
//	@contact.email               support_cept@indiapost.gov.in
//	@license.name                Apache 2.0
//	@license.url                 http://www.apache.org/licenses/LICENSE-2.0.html
//	@host                        localhost:8080
//	@BasePath                    /v1
//	@schemes                     http https
package main

import (
	"context"
	"pmdm/bootstrap"
	"pmdm/router"

	bootstrapper "gitlab.cept.gov.in/it-2.0-common/api-bootstrapper"
	"go.uber.org/fx"
)

func main() {
	app := bootstrapper.New().Options(
		bootstrap.Fxclient,
		bootstrap.Fxvalidator,
		fx.Invoke(router.Routes),
		bootstrap.FxHandler,
		bootstrap.FxRepo,
		bootstrap.FxMinio,
	)

	app.WithContext(context.Background()).Run()

}
