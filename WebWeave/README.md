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

4. Install dependencies:
   ```bash
   npm install
   ```

5. Start the development server:
   ```bash
   npm run dev
   ```

## Usage

1. Open the application in your browser.
2. Use the search bar to find specific tabs.
3. Add new tabs by entering URLs.
4. Drag and drop tabs to organize them into categories.
5. Use the export button to save your tab data.

### Example Usage

1. **Adding a New Tab**:
   - Enter the URL of the web page you want to add in the input field.
   - Click the "Add Tab" button.
   - The tab will be added to the list and categorized based on its content.

2. **Organizing Tabs**:
   - Drag and drop tabs between categories to organize them.
   - Use the search bar to filter tabs by keywords or URLs.

3. **Exporting Tab Data**:
   - Click the "Export" button to download a JSON file containing your tab data.
   - The exported data includes tab titles, URLs, summaries, and categories.

## Security Notes

- Never commit `.env` files to version control
- Keep your API keys private and secure
- Regularly rotate your API keys
- Use environment-specific `.env` files for different environments
- Monitor API usage and set up usage alerts
