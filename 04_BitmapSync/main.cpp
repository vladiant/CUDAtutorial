
#include <gtk/gtk.h>

extern "C" {
#include "generate_bitmap.h"
}

constexpr int dimension = 512;

struct UserData {
  guchar *pixels{nullptr};
  GtkImage *image{nullptr};
  GdkPixbuf *pixbuf{nullptr};
};

gboolean game_loop(GtkWidget *widget, GdkFrameClock *clock, gpointer data) {
  UserData *user_data = static_cast<UserData *>(data);
  GenerateBitmap(user_data->pixels, dimension);
  gtk_image_set_from_pixbuf(user_data->image, user_data->pixbuf);
  return 1;
}

int main() {
  gtk_init(nullptr, nullptr);

  UserData user_data;
  user_data.image = GTK_IMAGE(gtk_image_new());
  user_data.pixbuf =
      gdk_pixbuf_new(GDK_COLORSPACE_RGB, FALSE, 8, dimension, dimension);
  user_data.pixels = gdk_pixbuf_get_pixels(user_data.pixbuf);

  GtkWidget *window, *box;
  window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

  gtk_box_pack_start(GTK_BOX(box), GTK_WIDGET(user_data.image), TRUE, TRUE, 0);
  gtk_container_add(GTK_CONTAINER(window), box);
  g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), nullptr);

  gtk_widget_add_tick_callback(window, game_loop, &user_data, nullptr);
  gtk_widget_show_all(window);

  gtk_main();
}
