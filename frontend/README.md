# MOF CO2 Adsorption Predictor - Frontend

This is the React frontend for the MOF CO2 Adsorption Predictor.

## Installation

```bash
cd frontend
npm install
```

## Running the App

```bash
npm start
```

The app will open at `http://localhost:3000`

## Requirements

- Node.js 14+ and npm
- Flask API server running on `http://localhost:5000`

## Features

- Interactive form for inputting MOF properties
- Real-time prediction using the trained neural network
- Responsive design with gradient UI
- Error handling and loading states

## Development

To modify the app:
1. Edit components in `src/App.js`
2. Update styles in `src/index.css`
3. The app will hot-reload automatically

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` directory.
