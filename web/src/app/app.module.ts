import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { BrowserModule } from '@angular/platform-browser';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule, MatDialogModule, MatTabsModule } from '@angular/material';

import { AppComponent } from './app.component';
import { FooterComponent } from './components/footer/footer.component';
import { ProfileComponent } from './components/footer/profile/profile.component';
import { TestComponent } from './components/pages/test/test.component';
import { APP_ROUTES } from './app.routes';
import { TimelineComponent } from './components/pages/timeline/timeline.component';

@NgModule({
  declarations: [
    AppComponent,
    FooterComponent,
    ProfileComponent,
    TestComponent,
    TimelineComponent
  ],
  imports: [
    BrowserModule,
    RouterModule,
    APP_ROUTES,
    BrowserAnimationsModule,
    MatButtonModule,
    MatDialogModule,
    MatTabsModule
  ],
  entryComponents: [
    ProfileComponent
  ],
  providers: [
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
