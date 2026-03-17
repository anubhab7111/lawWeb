import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import lawyerRoutes from './routes/lawyers';
import chatRoutes from './routes/chat';
import authRoutes from './routes/auth';
import bookingRoutes from './routes/bookings'; // Added this import

// 1. Load environment variables
dotenv.config();

const app = express();
const PORT = parseInt(process.env.PORT || '5001', 10);

// 2. Middleware
app.use(cors());
app.use(express.json());

// 3. Request Logging (Helpful for debugging)
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
});

// 4. API Routes
app.use('/api/lawyers', lawyerRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/auth', authRoutes);
app.use('/api/bookings', bookingRoutes); // Added this line to register payment routes

// Log Python API connection
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000';
console.log(`📡 Python Chatbot API: ${PYTHON_API_URL}`);

// This will log only the cluster name, keeping your password safe
console.log("Checking Connection to:", process.env.MONGODB_URI?.split('@')[1] || "NOT FOUND - USING LOCALHOST");

// 5. Database Connection & Server Startup
const MONGODB_URI = process.env.MONGODB_URI;

const startServer = async () => {
    try {
        // Check if URI exists before trying to connect
        if (!MONGODB_URI) {
            throw new Error("MONGODB_URI is missing from your .env file!");
        }

        // Wait for MongoDB to connect
        await mongoose.connect(MONGODB_URI);
        console.log('✅ Success: Connected to MongoDB Atlas');

        // Start the Express server only AFTER the database is ready
        app.listen(PORT, '0.0.0.0', () => {
            console.log(`🚀 Server is running on http://localhost:${PORT}`);
        });

    } catch (err) {
        console.error('❌ CRITICAL ERROR: Database connection failed');
        console.error(err);
        
        // Force the process to exit so you don't have a "hanging" server
        process.exit(1);
    }
};

// Execute the startup
startServer();