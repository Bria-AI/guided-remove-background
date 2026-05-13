.PHONY: setup images run serve benchmark clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	uv sync

images: ## Download benchmark images from Pexels
	uv run python benchmark/fetch_images.py

run: images ## Run all 58 benchmark cases (requires API keys in .env)
	uv run python benchmark/runner.py --mode guided -v

serve: ## Start the live dashboard at http://localhost:8899
	@echo ""
	@echo "  Dashboard → http://localhost:8899/live.html"
	@echo "  Press Ctrl+C to stop"
	@echo ""
	@uv run python benchmark/feedback_server.py || true

benchmark: setup run serve ## Full pipeline: install → fetch images → run benchmark → open dashboard

grade: ## Auto-grade results with VLM
	uv run python benchmark/grader/run_grader.py

clean: ## Remove benchmark results (keeps images)
	rm -rf benchmark/results
