import mongoose from 'mongoose';

const BookingSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  lawyerId: { type: String, required: true },
  amount: { type: Number, required: true },
  // Updated: We use a generic transactionId for Braintree
  transactionId: { type: String, required: true }, 
  status: { 
    type: String, 
    enum: ['pending', 'confirmed', 'failed'], 
    default: 'pending' 
  },
  appointmentDate: { type: String },
  appointmentTime: { type: String },
  createdAt: { type: Date, default: Date.now }
});

export default mongoose.model('Booking', BookingSchema);