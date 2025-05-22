setup:
		@echo "Setting up the environment..."
		uv sync
		uv run pre-commit install
		echo "OPENAI_API_KEY=" > .env
		echo "GEMINI_API_KEY=" >> .env
		echo "ANTHROPIC_API_KEY=" >> .env
		@echo "Remember to fill .env"

activate:
		@echo "Activating virtual environment"
		source .venv/bin/activate
