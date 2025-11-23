import os
import logging
import tempfile

from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from image_result import detect_and_classify
from llm_text_result import get_text


load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is not set in environment variables (.env)")


# ---------------- Logging ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger("waste-bot")


# ---------------- Handlers ---------------- #

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! 👋\n\n"
        "Send me a photo of an item you aren’t sure how to dispose of.\n\n"
        "I will:\n"
        "• detect the object in the image,\n"
        "• highlight it and classify it (battery, biological, cardboard, clothes, glass, metal, trash, paper, plastic),\n"
        "• generate instructions on how to dispose of it properly.\n\n"
        "You can also use:\n"
        "/help — explanation\n"
        "/about — details about this bot\n"
    )

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Usage:\n"
        "1) Send a photo.\n"
        "2) Wait a moment.\n"
        "3) I’ll send:\n"
        "   • annotated image\n"
        "   • disposal instructions\n\n"
        "Commands:\n"
        "/start — greeting\n"
        "/help — help info\n"
        "/about — info about bot architecture\n"
    )

async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Waste Classifier Bot\n"
        "• Image detection — local PyTorch model (model.pth)\n"
        "• Text instructions — YandexGPT\n"
        "• Architecture — clean, modular, extensible\n"
        "• Language — English\n"
    )

async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    text = message.text

    # Ignore commands (commands go elsewhere)
    if text.startswith("/"):
        return  # command handler will catch it

    await message.reply_text(
        "This bot only works with photos. "
        "Your message is not a valid command."
    )

async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎤 Voice messages are not supported.\n\n"
        "Please send a photo of an item for waste classification or use a text command."
    )

async def video_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎥 Videos are not supported.\n\n"
        "Please send a photo of an item for waste classification or use a text command."
    )

async def video_note_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⭕ Video circles are not supported.\n\n"
        "Please send a photo of an item for waste classification or use a text command."
    )

async def sticker_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    sticker = message.sticker

    if not sticker:
        return

    if sticker.is_video:
        await message.reply_text(
            "🎬 Video stickers are not supported.\n\n"
            "Please send a static photo sticker or regular photo for waste classification."
        )
        return
    try:
        tmp_dir = tempfile.mkdtemp(prefix="waste_sticker_")
        input_path = os.path.join(tmp_dir, f"sticker_{sticker.file_unique_id}.png")

        tg_file = await context.bot.get_file(sticker.file_id)
        await tg_file.download_to_drive(custom_path=input_path)

        logger.info(f"Converted sticker to image at {input_path}")
        await process_image_file(update, context, input_path, "sticker")

    except Exception as e:
        logger.exception("Sticker processing error")
        await message.reply_text(f"❌ Error while processing sticker: {e}")

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    photo_list = message.photo

    if not photo_list:
        await message.reply_text("Please send a *photo*.")
        return

    photo = photo_list[-1]  # Берем фото наибольшего качества

    try:
        tmp_dir = tempfile.mkdtemp(prefix="waste_photo_")
        input_path = os.path.join(tmp_dir, f"photo_{photo.file_unique_id}.jpg")

        # Скачиваем фото (исправленная версия)
        tg_file = await context.bot.get_file(photo.file_id)
        await tg_file.download_to_drive(custom_path=input_path)

        logger.info(f"Downloaded user photo to {input_path}")

        # Используем единую функцию обработки
        await process_image_file(update, context, input_path, "photo")

    except Exception as e:
        logger.exception("Photo processing error")
        await message.reply_text(f"❌ Error during photo processing: {e}")

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик для изображений, отправленных как документы"""
    message = update.message
    document = message.document

    if not document:
        return

    # Получаем имя файла и расширение
    file_name = document.file_name or "unknown"
    file_extension = file_name.lower().split('.')[-1] if '.' in file_name else "no extension"
    
    # Разрешенные расширения
    allowed_extensions = ['jpg', 'jpeg', 'png']
    
    # Проверяем расширение файла или MIME type
    is_allowed_extension = file_extension in allowed_extensions
    is_image_mime = document.mime_type and document.mime_type.startswith('image/')
    
    if is_allowed_extension or is_image_mime:
        try:
            tmp_dir = tempfile.mkdtemp(prefix="waste_doc_")
            # Сохраняем с оригинальным расширением или используем jpg по умолчанию
            if file_extension in allowed_extensions:
                input_path = os.path.join(tmp_dir, f"doc_{document.file_unique_id}.{file_extension}")
            else:
                input_path = os.path.join(tmp_dir, f"doc_{document.file_unique_id}.jpg")

            # Скачиваем документ
            tg_file = await context.bot.get_file(document.file_id)
            await tg_file.download_to_drive(custom_path=input_path)

            logger.info(f"Downloaded image document '{file_name}' to {input_path}")

            # Используем единую функцию обработки
            await process_image_file(update, context, input_path, f"document: {file_name}")

        except Exception as e:
            logger.exception("Document image processing error")
            await message.reply_text(f"❌ Error processing image document: {e}")
    else:
        await message.reply_text(
            f"📄 Unsupported file type: .{file_extension}\n\n"
            "I can only process image files. Please send:\n"
            "✅ Supported formats: JPEG, JPG, PNG\n"
            "✅ Or send as: Photo directly in Telegram\n\n"
        )

async def unknown_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Unknown command. Try /help.")


# ---------------- Functions ---------------- #
async def process_image_file(update: Update, context: ContextTypes.DEFAULT_TYPE, input_path: str, source_type: str = "photo"):
    """Универсальная функция обработки изображений для всех типов входящих данных"""
    message = update.message
    
    try:
        # Отправляем сообщение о начале обработки
        progress_message = await message.reply_text(f"🔄 Processing {source_type}...")
        
        # Шаг 1: Детекция и классификация
        await context.bot.edit_message_text(
            chat_id=message.chat_id,
            message_id=progress_message.message_id,
            text="🔍 Analyzing objects..."
        )
        annotated_path, classifications = detect_and_classify(input_path)
        
        # Шаг 2: Генерация инструкций
        await context.bot.edit_message_text(
            chat_id=message.chat_id,
            message_id=progress_message.message_id,
            text="📝 Generating disposal instructions..."
        )
        text_instruction = get_text(classifications)
        
        # Удаляем сообщение о прогрессе
        await context.bot.delete_message(chat_id=message.chat_id, message_id=progress_message.message_id)
        
        # Отправляем результат
        with open(annotated_path, "rb") as f:
            caption_prefix = f"✅ Analysis complete! Detected: {', '.join(classifications) if classifications else 'No objects detected'}"
            
            if len(text_instruction) <= 500:
                full_caption = f"{caption_prefix}\n\n{text_instruction}"
                await message.reply_photo(photo=InputFile(f), caption=full_caption)
            else:
                await message.reply_photo(photo=InputFile(f), caption=caption_prefix)
                await message.reply_text(text_instruction)
                
    except Exception as e:
        # Пытаемся удалить сообщение о прогрессе в случае ошибки
        try:
            await context.bot.delete_message(chat_id=message.chat_id, message_id=progress_message.message_id)
        except:
            pass
        
        logger.exception(f"Error processing {source_type}")
        await message.reply_text(f"❌ Error processing {source_type}: {e}")

def main():
    logger.info("Starting bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("about", about_handler))

    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.Sticker.ALL, sticker_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_handler))
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    app.add_handler(MessageHandler(filters.VIDEO, video_handler))
    app.add_handler(MessageHandler(filters.VIDEO_NOTE, video_note_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))  

    app.add_error_handler(error_handler)

    logger.info("Bot setup complete, starting polling...")
    app.run_polling()

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Exception while handling an update: {context.error}")


if __name__ == "__main__":
    main()
