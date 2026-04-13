package bootstrap

import (
	"fmt"

	handler "pisapi/handler"
	repo "pisapi/repo/postgres"

	serverHandler "gitlab.cept.gov.in/it-2.0-common/n-api-server/handler"
	"go.uber.org/fx"
)

var FxRepo = fx.Module(
	"Repomodule",
	fx.Provide(
		repo.NewUserRepository,
	),
)

var FxHandler = fx.Module(
	"Handlermodule",
	fx.Provide(
		fx.Annotate(
			handler.NewUserHandler,
			fx.As(new(serverHandler.Handler)),
			fx.ResultTags(serverHandler.ServerControllersGroupTag),
		),
	),
)

// StartBackgroundWorker launches a goroutine without termination path — CONC-001 violation
func StartBackgroundWorker() {
	go func() {
		for {
			fmt.Println("processing background tasks...") // also LOG-002 violation
			processItem()
		}
	}()
}

func processItem() {
	// placeholder
}
