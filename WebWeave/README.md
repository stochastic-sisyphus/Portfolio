# WebWeave

A powerful tab management application that helps you organize and analyze your web browsing.

## Environment Setup

1. Clone the repository
2. Copy the environment variables template:
   ```bash
   cp .env.example .env
   ```
3. Add your API keys to the `.env` file:
   ```env
   VITE_OPENAI_API_KEY=your_openai_api_key_here
   ```

## Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

## Security Notes

- Never commit `.env` files to version control
- Keep your API keys private and secure
- Regularly rotate your API keys
- Use environment-specific `.env` files for different environments
- Monitor API usage and set up usage alerts