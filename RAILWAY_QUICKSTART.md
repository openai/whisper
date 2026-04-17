# Railway Deployment - Quick Start (5 Minutes)

Deploy your Farsi Transcriber to Railway in just 5 minutes! 🚀

---

## **What You'll Get**

✅ Your app live online
✅ Free $5/month credit
✅ 24/7 uptime
✅ Automatic scaling
✅ No credit card needed (free tier)

---

## **Step 1: Create Railway Account** (2 min)

1. Go to **https://railway.app**
2. Click **"Login with GitHub"**
3. Authorize with your GitHub account
4. Done! You get $5 free credit ✅

---

## **Step 2: Create Backend Service** (2 min)

1. Click **"Create New Project"**
2. Select **"GitHub Repo"**
3. Find your **whisper** fork
4. Railway auto-detects Python project
5. **Root Directory:** `farsi_transcriber_web/backend`
6. Click **Deploy**
7. **Wait 2-3 minutes** for deployment
8. **Copy the URL** that appears (e.g., `https://farsi-api-xxx.railway.app`)

---

## **Step 3: Create Frontend Service** (1 min)

1. In Railway project, click **"New Service"** → **"GitHub Repo"**
2. Select **whisper** again
3. **Root Directory:** `farsi_transcriber_web`
4. Click **Deploy**
5. **Wait 3-5 minutes** for build and deployment

---

## **Step 4: Connect Frontend to Backend** (Bonus step - 1 min)

1. In Railway, select **frontend** service
2. Go to **Variables**
3. Edit `VITE_API_URL` and paste your backend URL from Step 2
4. Click **Deploy** to redeploy with correct API URL

---

## **That's It! 🎉**

Your app is now live! Click the frontend service to see your live URL.

Example URLs:
- Frontend: `https://farsi-transcriber-prod.railway.app`
- Backend: `https://farsi-api-prod.railway.app`

---

## **Test Your App**

1. Click your frontend URL
2. Add a file
3. Click Transcribe
4. Wait for transcription
5. Export results

---

## **Detailed Guide**

For more details, see: `farsi_transcriber_web/RAILWAY_DEPLOYMENT.md`

---

## **Cost**

- **First 3 months:** FREE ($5/month credit)
- **After that:** ~$2-3/month for personal use
- Can upgrade to paid tier for more resources

---

## **Common Issues**

**"API connection failed"**
- Make sure backend URL is correct in frontend variables
- Redeploy frontend after updating API URL

**"Model not loaded"**
- Wait 1-2 minutes on first transcription
- Model downloads on first use

**"Build failed"**
- Check Railway logs for errors
- Ensure all files are committed

---

## **Support**

For detailed setup help, see: `farsi_transcriber_web/RAILWAY_DEPLOYMENT.md`

---

**Your Farsi Transcriber is now online!** Share the URL with anyone! 🌐
