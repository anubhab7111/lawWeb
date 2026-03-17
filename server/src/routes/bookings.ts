import express from 'express';
import braintree from 'braintree';
import Booking from '../models/Booking';

const router = express.Router();

/**
 * 1. Initialize Braintree Gateway (Lazy initialization)
 * This function ensures the gateway uses the latest process.env variables.
 */
const getGateway = () => {
  return new braintree.BraintreeGateway({
    environment: braintree.Environment.Sandbox,
    merchantId: process.env.BRAINTREE_MERCHANT_ID || 'r654tnw9cgxs2k27',
    publicKey: process.env.BRAINTREE_PUBLIC_KEY || 'hbyw2s7466q2py4m',
    privateKey: process.env.BRAINTREE_PRIVATE_KEY || 'd6c490c4b95de81b1956ff09d98c8b95'
  });
};

/**
 * 2. GET CLIENT TOKEN
 * This provides the frontend with the authorization needed to render the payment UI.
 */
router.get('/client_token', async (req, res) => {
  try {
    const gateway = getGateway();
    // Generates a one-time token for the frontend
    const response = await gateway.clientToken.generate({});
    res.send(response.clientToken); 
  } catch (err) {
    console.error("Braintree Token Error:", err);
    res.status(500).send("Braintree Authentication Failed. Check your API keys.");
  }
});

/**
 * 3. CHECKOUT (Charge the Nonce)
 * This processes the "nonce" (secure reference) received from the client.
 */
router.post('/checkout', async (req, res) => {
  try {
    const { amount, paymentMethodNonce, lawyerId, userId } = req.body;
    const gateway = getGateway();

    // Call Braintree to finalize the sale using the nonce
    const result = await gateway.transaction.sale({
      amount: amount.toString(),
      paymentMethodNonce: paymentMethodNonce,
      options: {
        submitForSettlement: true // Automatically captures the funds
      }
    });

    if (result.success) {
      // Save the confirmed booking with the unique Braintree Transaction ID
      const newBooking = new Booking({
        userId,
        lawyerId,
        amount,
        status: 'confirmed',
        transactionId: result.transaction.id 
      });
      
      await newBooking.save();
      
      console.log(`✅ Success: Payment settled for User ${userId}`);
      res.status(200).json({ status: 'success', transactionId: result.transaction.id });
    } else {
      // Handle cases where the processor declined the transaction
      console.error("❌ Braintree Transaction Failed:", result.message);
      res.status(400).json({ status: 'error', message: result.message });
    }
  } catch (err) {
    console.error("Checkout Error:", err);
    res.status(500).json({ message: "Internal Server Error during checkout" });
  }
});

/**
 * 4. GET USER BOOKINGS
 * Fetches confirmed appointments from your MongoDB Atlas database.
 */
router.get('/user-bookings/:userId', async (req, res) => {
  try {
    const bookings = await Booking.find({ 
      userId: req.params.userId, 
      status: 'confirmed' 
    }).sort({ createdAt: -1 }); 
    
    res.json(bookings);
  } catch (error) {
    console.error("Fetch Bookings Error:", error);
    res.status(500).json({ message: "Fetch failed" });
  }
});

export default router;